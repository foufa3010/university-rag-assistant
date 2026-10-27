import time
from eval.run_eval import read_cases
from src.retrieval import HybridRetriever
from eval.metrics import recall_at_k
def main():
    r=HybridRetriever(use_reranker=False); cases=read_cases()
    for mode in ("keyword","vector","hybrid"):
        scores=[]; times=[]
        for c in cases:
            start=time.perf_counter(); ps=r.search(c["question"],top_k=5,mode=mode); times.append((time.perf_counter()-start)*1000)
            scores.append(recall_at_k([p["chunk_id"] for p in ps],c.get("relevant_chunk_ids",[]),5))
        n=max(1,len(cases)); print(f"{mode:8} Recall@5={sum(scores)/n:.3f} mean_search_ms={sum(times)/n:.1f}")
    print("To measure hybrid+reranker, instantiate HybridRetriever(use_reranker=True) and search mode=hybrid.")
if __name__=="__main__":main()
