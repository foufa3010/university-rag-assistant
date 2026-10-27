# Design Document

## Corpus and users
Target users are students and academic advisors asking about course prerequisites, degree completion, credit minimums, and academic policies. Included source files are synthetic examples. For a real deployment, record authoritative source, version date, owner, and review status; exclude private student data.

## Metadata and chunking
Every chunk carries document ID, title, source filename, date, author, section, and chunk index. Metadata fields absent from source documents remain null rather than being invented. Baseline chunking is paragraph-aware at 180 words with 35-word overlap. Long paragraphs are split into overlapping windows. Evaluate 100/20, 180/35, and 300/50 on the same reviewed set; select by retrieval recall, citation quality, and latency rather than intuition.

## Index choice
Normalized embeddings are saved in a flat NumPy matrix; dot product is cosine similarity. This is transparent, portable, and suitable for a small corpus. Search is O(N), so use FAISS, Qdrant, or PostgreSQL pgvector if scale/latency requires approximate nearest-neighbor search. BM25 is rebuilt at startup from the chunk text and catches exact identifiers such as CS301.

## Retrieval and reranking
BM25 and vector rankings are fused with reciprocal-rank fusion (RRF), then a cross-encoder reranks a limited candidate set. Metadata filters are applied before ranking. Compare keyword-only, vector-only, hybrid, and hybrid+reranker separately. Reranker scores are ranking scores, not calibrated confidence.

## Generation and citations
The optional Ollama model is instructed to answer only from retrieved passages and cite chunk IDs for every factual claim. If the local model is unavailable, the fallback quotes the highest-ranked passage with a citation. No retrieved passages yields “I don't know”. For production, add claim-to-citation validation and calibrate abstention on a held-out set.

## Known failure cases
- Tables can lose row/column relationships when flattened.
- Prerequisites and exceptions can be in different chunks or documents.
- Cross-references may require following a section pointer that retrieval misses.
- Conflicting policy versions and missing dates need explicit version handling.
- BM25 tokenization may mishandle unusual course IDs; dense retrieval may overlook exact codes.
- Compound questions may require evidence from several passages.
- A plausible answer can still be unsupported; citations alone do not prove faithfulness.
- Synthetic data and repeated question templates can inflate evaluation scores.
