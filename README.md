# docguard-ai

Built this after dealing with silent model degradation at ADM Northfield — document AI pipelines would quietly drop accuracy and nobody noticed for weeks. This adds an evaluation and governance layer so that doesn't happen.

Processes financial, healthcare, insurance, and legal documents through a multi-layer pipeline: Claude classifies and extracts entities, a RAG engine checks against known fraud patterns, a second Claude call judges the output quality, and anything low-confidence routes to a human review queue. Reviewer decisions feed back into the golden set.

## layers

```
upload → classify (claude tool use) → RAG verify (chromadb) → judge score → review queue if needed
                                                                    ↑
                                              accepted reviews feed back here (rlhf loop)
```

## setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY
python utils/models.py
uvicorn api.main:app --reload
```

## endpoints

```
POST /documents/process          upload PDF/DOCX/TXT, get back scores + risk
GET  /documents/{id}             doc details + latest eval
GET  /review-queue               pending human reviews
POST /review-queue/{id}/decide   accept/reject/correct (accepted → golden set)
GET  /governance/stats           quality metrics, window configurable
GET  /governance/drift-alerts    active drift alerts
POST /governance/golden-sets     add example manually
```

## env vars

```
ANTHROPIC_API_KEY       required
CONFIDENCE_THRESHOLD    0.75  (below this → review queue)
DRIFT_ALERT_THRESHOLD   0.10  (drop that triggers alert)
DRIFT_WINDOW_DAYS       7
DATABASE_URL            defaults to sqlite locally
```

## notes

- embeddings use TF-IDF locally to avoid model downloads — swap `LocalEmbeddings` in `retrieval_engine.py` for openai before prod
- judge eval adds ~1-2s per doc, not great for high-throughput batch, might move it async
- drift detection needs at least 5 evaluations per category before it kicks in
- no auth on the API yet

## tech

python, anthropic claude, langchain, chromadb, fastapi, sqlalchemy, sqlite/postgres
