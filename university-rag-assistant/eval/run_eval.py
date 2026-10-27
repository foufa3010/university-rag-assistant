import json,time,statistics
from pathlib import Path
from src.retrieval import HybridRetriever
from src.answering import answer_question
from eval.metrics import token_f1,recall_at_k
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data/eval/questions.jsonl"
def read_cases(): return [json.loads(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]
def main():
    r=HybridRetriever(); cases=read_cases(); rows=[]
    for c in cases:
        start=time.perf_counter(); passages=r.search(c["question"],top_k=5,mode="hybrid")
        ms=(time.perf_counter()-start)*1000; result=answer_question(c["question"],passages,use_llm=False)
        ids=[p["chunk_id"] for p in passages]
        rows.append((recall_at_k(ids,c.get("relevant_chunk_ids",[]),5),token_f1(result["answer"],c["answer"]),bool(result["citations"]),ms))
    n=max(1,len(rows)); lat=sorted(x[3] for x in rows)
    print(f"Cases: {len(rows)}")
    print(f"Recall@5: {sum(x[0] for x in rows)/n:.3f}")
    print(f"Answer token F1 (rough extractive proxy): {sum(x[1] for x in rows)/n:.3f}")
    print(f"Citation presence: {sum(x[2] for x in rows)/n:.3f}")
    print(f"Retrieval mean/p50/p95 ms: {statistics.mean(lat):.1f}/{lat[int(.50*(n-1))]:.1f}/{lat[int(.95*(n-1))]:.1f}")
    print("Run on your machine; these are measured results, not prefilled claims.")
if __name__=="__main__": main()
