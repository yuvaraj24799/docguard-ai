import os, sys, tempfile
from datetime import datetime
from typing import Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.document_agent import process_document
from rag.retrieval_engine import verify_with_rag, seed_knowledge_base
from evaluation.judge import evaluate_document, get_stats, add_to_golden_set
from utils.models import get_session, Document, Evaluation, ReviewQueueItem, ModelDriftAlert, ReviewAction, DocumentStatus, create_tables

load_dotenv()
create_tables()
seed_knowledge_base()

app = FastAPI(title="DocGuard AI", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ReviewDecision(BaseModel):
    action: str
    correction: Optional[dict] = None
    notes: Optional[str] = None

class GoldenEntry(BaseModel):
    category: str
    input_text: str
    expected_output: dict
    source: str = "manual"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/documents/process")
async def process(
    file: UploadFile = File(...),
    mode: str = Query("realtime", enum=["realtime", "batch"])
):
    if Path(file.filename).suffix.lower() not in {".pdf", ".docx", ".txt"}:
        raise HTTPException(400, "pdf/docx/txt only")

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
        content = await file.read()
        if len(content) > 10_000_000:
            raise HTTPException(400, "max 10mb")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        l1 = process_document(tmp_path, mode)
        if l1.get("status") == "failed":
            raise HTTPException(500, l1.get("error"))

        session = get_session()
        doc = session.query(Document).get(l1["document_id"])
        session.close()

        l2 = verify_with_rag(doc.raw_text or "", l1["structured_output"], l1["category"])
        l3 = evaluate_document(l1["document_id"], l2)

        return {
            "document_id": l1["document_id"],
            "filename": file.filename,
            "category": l1["category"],
            "confidence": l1["confidence"],
            "output": l1["structured_output"],
            "verification": {
                "risk_level": l2["risk_level"],
                "report": l2["verification_report"][:400] + "...",
                "fraud_matches": l2["fraud_pattern_matches"]
            },
            "eval": {
                "overall": l3["scores"]["overall"],
                "correctness": l3["scores"]["correctness"],
                "faithfulness": l3["scores"]["faithfulness"],
                "hallucination_risk": l3["scores"]["hallucination_risk"],
                "reasoning": l3["judge_reasoning"]
            },
            "flagged": l3["flagged_for_review"],
            "flag_reason": l3.get("flag_reason"),
            "drift": l3.get("drift_detected", False),
            "latency_ms": l1.get("latency_ms")
        }
    finally:
        os.unlink(tmp_path)


@app.get("/documents/{doc_id}")
def get_doc(doc_id: str):
    session = get_session()
    try:
        doc = session.query(Document).get(doc_id)
        if not doc:
            raise HTTPException(404)
        ev = session.query(Evaluation).filter_by(document_id=doc_id).order_by(Evaluation.created_at.desc()).first()
        return {
            "id": doc.id, "filename": doc.filename, "category": doc.document_category,
            "status": doc.status, "output": doc.structured_output,
            "eval": {"score": ev.overall_score, "flagged": ev.flagged_for_review, "drift": ev.drift_detected} if ev else None,
            "created": doc.created_at.isoformat()
        }
    finally:
        session.close()


@app.get("/review-queue")
def queue(limit: int = 20):
    session = get_session()
    try:
        items = session.query(ReviewQueueItem).filter_by(status="pending").order_by(ReviewQueueItem.created_at.desc()).limit(limit).all()
        return {"count": len(items), "items": [{
            "id": i.id, "doc_id": i.document_id, "score": i.confidence_score,
            "reason": i.flag_reason, "output": i.original_output, "at": i.created_at.isoformat()
        } for i in items]}
    finally:
        session.close()


@app.post("/review-queue/{item_id}/decide")
def decide(item_id: str, d: ReviewDecision):
    session = get_session()
    try:
        item = session.query(ReviewQueueItem).get(item_id)
        if not item:
            raise HTTPException(404)

        item.status = "reviewed"
        item.reviewer_action = ReviewAction(d.action)
        item.reviewer_correction = d.correction
        item.reviewer_notes = d.notes
        item.reviewed_at = datetime.utcnow()

        # accepted → golden set → closes the rlhf loop
        if d.action == "accepted":
            doc = session.query(Document).get(item.document_id)
            if doc:
                add_to_golden_set(doc.document_category or "unknown", (doc.raw_text or "")[:500], item.original_output or {})
                item.added_to_golden_set = True

        session.commit()
        return {"ok": True, "action": d.action, "golden": item.added_to_golden_set}
    finally:
        session.close()


@app.get("/governance/stats")
def stats(category: Optional[str] = None, days: int = Query(7, ge=1, le=90)):
    return get_stats(category, days)


@app.get("/governance/drift-alerts")
def alerts(resolved: bool = False):
    session = get_session()
    try:
        rows = session.query(ModelDriftAlert).filter_by(resolved=resolved).order_by(ModelDriftAlert.created_at.desc()).limit(50).all()
        return {"count": len(rows), "alerts": [{
            "id": a.id, "category": a.document_category, "metric": a.metric_name,
            "baseline": a.baseline_value, "current": a.current_value,
            "drop": a.drift_magnitude, "at": a.created_at.isoformat()
        } for a in rows]}
    finally:
        session.close()


@app.post("/governance/golden-sets")
def add_golden(e: GoldenEntry):
    gid = add_to_golden_set(e.category, e.input_text, e.expected_output, e.source)
    return {"id": gid, "ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
