# Week 8 Optimization Report

## Target
Reduce request cost or p95 latency by at least 50%, while losing no more than 2 absolute percentage points of answer accuracy. Keep the same frozen evaluation set and hardware for before/after comparisons.

## Measurements (fill after running)
| Metric | Baseline | Optimized | Difference |
|---|---:|---:|---:|
| Recall@5 | TODO | TODO | TODO |
| Answer correctness | TODO | TODO | TODO |
| Citation accuracy | TODO | TODO | TODO |
| p50 end-to-end latency | TODO | TODO | TODO |
| p95 end-to-end latency | TODO | TODO | TODO |
| Cost/request | TODO | TODO | TODO |

## Candidates to test
1. Send only top 3–4 passages to generation rather than 5–10.
2. Reduce reranker candidate pool from 15 to 8.
3. Cache normalized repeated questions and retrieved passage IDs (with index-version invalidation).
4. Skip reranking only when a calibrated retrieval-confidence rule is met.
5. Compare a smaller embedding/reranker/generation model.
6. Batch embeddings offline and avoid recomputing them for unchanged documents.

## Protocol
Run the exact same 100+ reviewed cases at least three times; report median/p95 and warm/cold cache conditions. Record CPU/GPU, RAM, model versions, chunk settings, and corpus size. Do not claim the target was met until measured. Local inference removes hosted token fees but not compute, electricity, or maintenance costs. At 100× traffic, estimate concurrent capacity and queueing from measured throughput.
