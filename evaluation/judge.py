import os
import json
import statistics
from datetime import datetime, timedelta
from typing import Optional
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv
from utils.models import (
    Evaluation, GoldenSet, ModelDriftAlert, ReviewQueueItem,
    Document, DocumentStatus, get_session
)

load_dotenv()

_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.75))
DRIFT_THRESHOLD = 0.10
DRIFT_WINDOW = 7

# explicit about the scale — claude was returning 0-10 before
_JUDGE_TOOL = [{
    "name": "score_output",
    "description": "Score LLM output quality. ALL scores must be decimal values between 0.0 and 1.0. Do not use a 0-10 scale.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correctness_score": {
                "type": "number",
                "description": "MUST be between 0.0 and 1.0. Is the output factually correct based on source?"
            },
            "faithfulness_score": {
                "type": "number",
                "description": "MUST be between 0.0 and 1.0. Does output avoid hallucinating facts not in source?"
            },
            "completeness_score": {
                "type": "number",
                "description": "MUST be between 0.0 and 1.0. Does output cover all key aspects?"
            },
            "hallucination_risk_score": {
                "type": "number",
                "description": "MUST be between 0.0 and 1.0. Higher = more hallucination risk detected."
            },
            "reasoning": {"type": "string"},
            "specific_issues": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["correctness_score", "faithfulness_score", "completeness_score", "hallucination_risk_score", "reasoning"]
    }
}]


def _clamp(val, lo=0.0, hi=1.0):
    """safety net — clamp any score that slips out of range"""
    if val is None:
        return 0.5
    v = float(val)
    # if claude returned 0-10 scale, normalize it
    if v > 1.0:
        v = v / 10.0
    return max(lo, min(hi, round(v, 4)))


def _judge(source: str, output: dict, category: str) -> dict:
    resp = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="""You are a quality evaluator for enterprise AI systems.
Score outputs on a 0.0 to 1.0 scale ONLY. Never use scores above 1.0.
Examples: good correctness = 0.85, poor faithfulness = 0.30, high hallucination risk = 0.70.
Use the score_output tool.""",
        tools=_JUDGE_TOOL,
        messages=[{
            "role": "user",
            "content": f"Evaluate this {category} output.\n\nSource:\n{source[:1500]}\n\nOutput:\n{json.dumps(output)[:1500]}\n\nScore all dimensions 0.0-1.0 using score_output tool."
        }]
    )

    for b in resp.content:
        if b.type == "tool_use" and b.name == "score_output":
            s = b.input
            # clamp everything just in case
            cor = _clamp(s.get("correctness_score"))
            fai = _clamp(s.get("faithfulness_score"))
            com = _clamp(s.get("completeness_score"))
            hal = _clamp(s.get("hallucination_risk_score"))
            overall = round(cor*0.35 + fai*0.35 + com*0.20 + (1-hal)*0.10, 4)
            return {
                "correctness_score": cor,
                "faithfulness_score": fai,
                "completeness_score": com,
                "hallucination_risk_score": hal,
                "overall": overall,
                "reasoning": s.get("reasoning", ""),
                "specific_issues": s.get("specific_issues", [])
            }

    # TODO: retry logic here
    logger.warning("judge skipped tool call, using fallback")
    return {
        "correctness_score": 0.5, "faithfulness_score": 0.5,
        "completeness_score": 0.5, "hallucination_risk_score": 0.5,
        "overall": 0.5, "reasoning": "eval failed", "specific_issues": []
    }


def _golden_compare(output: dict, category: str, session) -> Optional[dict]:
    examples = session.query(GoldenSet).filter_by(document_category=category).limit(3).all()
    if not examples:
        return None
    ex_str = "\n".join(f"Ex{i+1}: {json.dumps(e.expected_output)[:200]}" for i, e in enumerate(examples))
    try:
        resp = _client.messages.create(
            model="claude-sonnet-4-6", max_tokens=200,
            messages=[{"role": "user", "content": f"Compare to golden set.\n\nGenerated: {json.dumps(output)[:400]}\n\nGolden:\n{ex_str}\n\nJSON: {{\"similarity\": 0.8, \"deviations\": []}}"}]
        )
        txt = resp.content[0].text
        return json.loads(txt[txt.find("{"):txt.rfind("}")+1])
    except:
        return None


def _check_drift(category: str, score: float, session) -> Optional[dict]:
    cutoff = datetime.utcnow() - timedelta(days=DRIFT_WINDOW)
    recent = session.query(Evaluation).join(Document).filter(
        Document.document_category == category,
        Evaluation.created_at >= cutoff
    ).all()

    if len(recent) < 5:
        return None

    baseline = statistics.mean(e.overall_score for e in recent if e.overall_score)
    drop = baseline - score

    if drop >= DRIFT_THRESHOLD:
        session.add(ModelDriftAlert(
            document_category=category, metric_name="overall",
            baseline_value=round(baseline, 4), current_value=round(score, 4),
            drift_magnitude=round(drop, 4), alert_threshold=DRIFT_THRESHOLD
        ))
        logger.warning(f"drift alert | {category} | {baseline:.3f} → {score:.3f} (drop={drop:.3f})")
        return {"drift_detected": True, "baseline": round(baseline, 4), "current": round(score, 4), "drop": round(drop, 4)}

    return {"drift_detected": False}


def evaluate_document(document_id: str, verification_result: dict) -> dict:
    session = get_session()
    try:
        doc = session.query(Document).get(document_id)
        if not doc:
            raise ValueError(f"doc {document_id} not found")

        scores = _judge(doc.raw_text or "", doc.structured_output or {}, doc.document_category or "unknown")
        overall = scores.get("overall", 0.5)

        golden = _golden_compare(doc.structured_output or {}, doc.document_category or "unknown", session)
        drift = _check_drift(doc.document_category or "unknown", overall, session)
        drifted = drift.get("drift_detected", False) if drift else False

        flagged = overall < CONF_THRESHOLD
        issues = scores.get("specific_issues", [])
        reason = f"score {overall:.2f}: {', '.join(issues[:2])}" if flagged and issues else (f"low score {overall:.2f}" if flagged else None)

        ev = Evaluation(
            document_id=document_id,
            correctness_score=scores.get("correctness_score"),
            faithfulness_score=scores.get("faithfulness_score"),
            completeness_score=scores.get("completeness_score"),
            hallucination_risk_score=scores.get("hallucination_risk_score"),
            overall_score=overall,
            judge_reasoning={"reasoning": scores.get("reasoning"), "issues": issues},
            golden_set_comparison=golden,
            sources_cited=verification_result.get("sources_cited", []),
            confidence=overall,
            flagged_for_review=flagged,
            flag_reason=reason,
            drift_detected=drifted,
            drift_magnitude=drift.get("drop") if drift else None
        )
        session.add(ev)

        if flagged:
            session.add(ReviewQueueItem(
                document_id=document_id, evaluation_id=ev.id,
                original_output=doc.structured_output, confidence_score=overall, flag_reason=reason
            ))
            doc.status = DocumentStatus.REVIEW_REQUIRED

        session.commit()
        logger.info(f"eval {document_id}: {overall:.2f} flagged={flagged}")

        return {
            "evaluation_id": ev.id, "document_id": document_id,
            "scores": {
                "correctness": scores.get("correctness_score"),
                "faithfulness": scores.get("faithfulness_score"),
                "completeness": scores.get("completeness_score"),
                "hallucination_risk": scores.get("hallucination_risk_score"),
                "overall": overall
            },
            "judge_reasoning": scores.get("reasoning"),
            "flagged_for_review": flagged, "flag_reason": reason,
            "drift_detected": drifted, "golden_set_comparison": golden
        }

    except Exception as e:
        session.rollback()
        logger.error(f"eval failed {document_id}: {e}")
        raise
    finally:
        session.close()


def add_to_golden_set(category: str, input_text: str, expected_output: dict, source="human_review") -> str:
    session = get_session()
    try:
        g = GoldenSet(document_category=category, input_text=input_text, expected_output=expected_output, source=source)
        session.add(g)
        session.commit()
        return g.id
    finally:
        session.close()


def get_stats(category: Optional[str]=None, days: int=7) -> dict:
    session = get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        q = session.query(Evaluation).filter(Evaluation.created_at >= cutoff)
        if category:
            q = q.join(Document).filter(Document.document_category == category)
        evs = q.all()
        if not evs:
            return {"count": 0, "msg": "no data yet"}
        sc = [e.overall_score for e in evs if e.overall_score]
        return {
            "count": len(evs), "avg": round(statistics.mean(sc), 4) if sc else 0,
            "min": round(min(sc), 4) if sc else 0, "max": round(max(sc), 4) if sc else 0,
            "flagged": sum(1 for e in evs if e.flagged_for_review),
            "drift_alerts": sum(1 for e in evs if e.drift_detected),
            "days": days
        }
    finally:
        session.close()
