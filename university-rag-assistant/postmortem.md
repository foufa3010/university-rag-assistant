# Postmortem

Complete this after actually running the repository. Do not invent problems, metrics, or lessons.

## Incident / issue observed
- Symptom: TODO
- Root cause: TODO
- User or metric impact: TODO
- Fix and verification: TODO

## What we learned
- Which chunk size/overlap gave the best measured recall and why?
- Which questions failed and was the cause lexical, semantic, metadata, or cross-reference related?
- Did reranking improve relevance enough to justify its latency?
- Were any answer claims unsupported by their citations?
- What should change in source document preparation next time?

## Follow-up
| Action | Priority | Verification |
|---|---|---|
| Replace synthetic corpus with approved source documents | High | Owner approves corpus list |
| Expert-review 100+ questions and gold passages | High | Gold IDs manually checked |
| Measure p50/p95 on deployment hardware | High | Reproducible benchmark output |
| Add regression cases for conflicting rules | Medium | Tests pass |
