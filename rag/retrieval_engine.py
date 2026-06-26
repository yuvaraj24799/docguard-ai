import os
import json
from typing import Optional
import numpy as np
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from anthropic import Anthropic
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# using tfidf locally so we don't need to download models in dev
# switching to openai text-embedding-3-small before going to prod
# had issues with onnx model checksums in the container environment
class LocalEmbeddings(EmbeddingFunction):
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=512, ngram_range=(1,2), sublinear_tf=True)
        self._docs = []

    def __call__(self, input: Documents) -> Embeddings:
        self._docs.extend(input)
        self.vec.fit(list(set(self._docs)))
        mat = self.vec.transform(input).toarray()
        # pad to 128 dims
        if mat.shape[1] < 128:
            mat = np.hstack([mat, np.zeros((mat.shape[0], 128 - mat.shape[1]))])
        else:
            mat = mat[:, :128]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return (mat / np.where(norms==0, 1, norms)).tolist()


_chroma = chromadb.Client()
_emb = LocalEmbeddings()


def _col(name: str):
    try:
        return _chroma.get_collection(name, embedding_function=_emb)
    except:
        return _chroma.create_collection(name, embedding_function=_emb)


def chunk(text: str, size=512, overlap=50) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size-overlap)]


def index_doc(doc_id: str, text: str, category: str, col="fraud_patterns") -> int:
    c = _col(col)
    chunks = chunk(text)
    c.upsert(
        documents=chunks,
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        metadatas=[{"doc_id": doc_id, "cat": category, "idx": i} for i in range(len(chunks))]
    )
    return len(chunks)


def search(query: str, col="fraud_patterns", n=5, cat: Optional[str]=None) -> list[dict]:
    c = _col(col)
    if c.count() == 0:
        return []
    try:
        res = c.query(
            query_texts=[query],
            n_results=min(n, c.count()),
            where={"cat": cat} if cat else None,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {"rank": i+1, "text": d, "meta": m, "score": round(max(0, 1-dist), 4)}
            for i, (d, m, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))
        ]
    except Exception as e:
        logger.warning(f"search failed: {e}")
        return []


def rerank(query: str, hits: list[dict]) -> list[dict]:
    if len(hits) <= 1:
        return hits
    passages = "\n".join(f"[{i+1}] {h['text'][:250]}" for i, h in enumerate(hits))
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=128,
            messages=[{"role": "user", "content": f"Rerank by relevance to: {query}\n\n{passages}\n\nJSON array only, e.g. [2,1,3]"}]
        )
        txt = resp.content[0].text
        order = json.loads(txt[txt.find("["):txt.find("]")+1])
        reranked = [hits[i-1] for i in order if 0 < i <= len(hits)]
        for i, h in enumerate(reranked):
            h["rank"] = i+1
        return reranked
    except:
        return hits


def verify_with_rag(document_text: str, structured_output: dict, category: str) -> dict:
    q = f"{category} {structured_output.get('summary','')} {' '.join(structured_output.get('risk_indicators',[]))}"

    fraud_hits = rerank(q, search(q, "fraud_patterns", n=5))
    comp_hits = search(q, "compliance_rules", n=3)

    ctx = []
    if fraud_hits:
        ctx.append("Fraud patterns:\n" + "\n".join(f"[S{i+1}] {h['text'][:200]}" for i, h in enumerate(fraud_hits[:3])))
    if comp_hits:
        ctx.append("Compliance:\n" + "\n".join(f"[R{i+1}] {h['text'][:150]}" for i, h in enumerate(comp_hits[:2])))

    context = "\n\n".join(ctx) or "no relevant patterns found"

    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=800,
        system="Document verifier. Cite sources [S1], [R1] etc. Conservative risk assessment.",
        messages=[{"role": "user", "content": f"Verify this {category}.\n\nSummary: {structured_output.get('summary')}\nFlags: {structured_output.get('risk_indicators',[])}\n\nContext:\n{context}\n\nRisk level + matched patterns + verification status."}]
    )

    report = resp.content[0].text
    lvl = "high" if "high" in report.lower() else "medium" if "medium" in report.lower() else "low"

    return {
        "verification_report": report,
        "risk_level": lvl,
        "sources_cited": [h["meta"] for h in fraud_hits[:3]],
        "compliance_flags": [h["text"][:80] for h in comp_hits if h["score"] > 0.6],
        "fraud_pattern_matches": sum(1 for h in fraud_hits if h["score"] > 0.6),
        "used_context": bool(ctx)
    }


def seed_knowledge_base():
    fraud = _col("fraud_patterns")
    comp = _col("compliance_rules")

    fp = [
        "Invoice amounts that don't match purchase order totals indicate potential fraud",
        "Duplicate invoice numbers across different vendors suggest billing fraud",
        "Round number invoices ending in .00 are statistically more likely to be fraudulent",
        "Vendor addresses matching employee addresses indicate insider fraud risk",
        "Claims filed shortly after policy inception often indicate fraud",
        "Multiple claims from same IP address suggest coordinated fraud ring",
        "Healthcare claims for procedures not covered by diagnosis code indicate upcoding",
        "Loan applications with income figures inconsistent with employment history",
        "Financial statements with unusual revenue spikes at quarter end",
        "Contracts with above-market rates to related parties indicate self-dealing"
    ]
    cr = [
        "SOX requires dual approval for transactions over $10,000",
        "HIPAA requires PHI de-identification before processing",
        "AML requires SAR filing for suspicious transactions over $5,000",
        "GDPR requires explicit consent for personal data processing",
        "Insurance claims must be filed within 30 days of incident",
        "Financial statements must follow GAAP revenue recognition"
    ]

    fraud.upsert(documents=fp, ids=[f"f{i}" for i in range(len(fp))], metadatas=[{"cat":"fraud"} for _ in fp])
    comp.upsert(documents=cr, ids=[f"c{i}" for i in range(len(cr))], metadatas=[{"cat":"compliance"} for _ in cr])
    logger.info(f"seeded: {len(fp)} fraud patterns, {len(cr)} compliance rules")
