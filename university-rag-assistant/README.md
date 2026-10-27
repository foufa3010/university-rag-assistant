# University RAG Assistant

A reproducible Retrieval-Augmented Generation (RAG) project for university course catalogues and degree rules. Includes metadata-aware chunking, dense retrieval, BM25, hybrid fusion, cross-encoder reranking, grounded answers with citations, abstention, 110 evaluation questions, ablation scripts, tests, a Streamlit UI, and a FastAPI endpoint.

> **Demo data only:** documents are synthetic examples, not real university policy. Replace them with approved official documents before real students rely on the system. Never upload student records or secrets.

## Architecture

Offline: load `.txt`/`.md` files and metadata → paragraph-aware chunking → embeddings → flat NumPy vector index + BM25 keyword index.

Online: normalize/rewrite query → optional metadata filters → keyword + vector retrieval → reciprocal-rank fusion → optional cross-encoder reranking → evidence-only answer with citations or “I don't know”.

## Setup and run

Python 3.10 or 3.11 recommended. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.build_index
streamlit run app/streamlit_app.py
```

First run may download embedding/reranker models. They run locally afterward. Optional local LLM: install Ollama and run `ollama pull llama3.2`; set `OLLAMA_MODEL` if using another model. Without Ollama, the system returns an extractive evidence-based fallback.

Optional API: `uvicorn src.api:app --host 127.0.0.1 --port 8000`, then open `/docs`.

## Evaluation

```bash
python -m eval.populate_gold_ids
python -m eval.run_eval
python -m eval.ablation
pytest -q
```

Review the generated gold passage IDs manually before citing metrics. The included 110 questions are synthetic and repeated in varied wording; replace/extend them with expert-reviewed questions. Metrics are not pre-filled because they must be measured on your machine and actual corpus.

## Design decisions

- **Vector storage:** normalized embeddings in a flat NumPy matrix. It is transparent, portable, and appropriate for a small course catalogue. Search is O(N); use FAISS/Qdrant/Postgres pgvector if corpus scale makes that too slow.
- **Keyword index:** BM25 catches exact course IDs and policy terms that semantic search can miss.
- **Hybrid fusion:** reciprocal-rank fusion avoids comparing incomparable raw BM25 and cosine scores.
- **Reranker:** cross-encoder reranks a small candidate set; it may improve relevance but adds latency.
- **Chunking:** default 180 words / 35-word overlap, paragraph-aware; compare 100/20, 180/35, 300/50 on the same eval set.
- **Abstention:** retrieval score is not a calibrated probability. Tune thresholds on held-out examples; do not treat the demo threshold as production-ready.

## Results

Run evaluation scripts and add measured Recall@5, answer correctness, citation accuracy, abstention accuracy, p50/p95 end-to-end latency, and request cost here. Include hardware, corpus size, model versions, and warm/cold status. Do not invent numbers.

## Cost model

Local inference has no hosted per-token charge, but uses compute, electricity, and maintenance. For hosted models: `request_cost = input_tokens * input_price + output_tokens * output_price`. Estimate load at 100× using measured request volume and p95 latency; note capacity constraints.

## Deployment

Deploy Streamlit on a trusted host or run locally. For real students, add authentication, HTTPS, rate limits, audit logging, document versioning, retention rules, and a clear disclaimer that the chatbot does not make official graduation decisions.

## Known limitations

Tables may lose row/column structure when converted to text; prerequisite cross-references can be in different chunks; policy exceptions may conflict with general rules; source dates may be stale; course-code tokenization can affect retrieval; multi-part questions may need several passages; reranker scores are not probabilities; extractive fallback can be incomplete. Validate against official data and advisors.

## Contributions

1. Open an issue describing the change.
2. Create a branch and implement it with tests.
3. Add/update reviewed evaluation questions.
4. Run `pytest -q`, `python -m eval.run_eval`, and `python -m eval.ablation`.
5. Update design/optimization documentation with measured effects.
6. Submit a pull request with test and metric results.
