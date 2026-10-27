import json
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import INDEX_DIR,EMBEDDING_MODEL,RERANKER_MODEL,TOP_K_KEYWORD,TOP_K_VECTOR,RRF_K
from src.indexing import tokenize

class HybridRetriever:
    def __init__(self,index_dir:Path=INDEX_DIR,use_reranker:bool=True):
        cp,vp=index_dir/"chunks.json",index_dir/"vectors.npy"
        if not cp.exists() or not vp.exists(): raise FileNotFoundError("Index missing; run python -m src.build_index")
        self.chunks=json.loads(cp.read_text(encoding="utf-8")); self.vectors=np.load(vp)
        manifest=json.loads((index_dir/"manifest.json").read_text(encoding="utf-8"))
        self.model=SentenceTransformer(manifest.get("embedding_model",EMBEDDING_MODEL))
        self.bm25=BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        # Small in-process LRU-like cache: repeated questions avoid embedding/search/rerank work.
        self._cache = {}
        self._cache_limit = 256
        self.reranker=None
        if use_reranker:
            try: self.reranker=CrossEncoder(RERANKER_MODEL)
            except Exception as exc: print(f"Reranker unavailable; using RRF order: {exc}")
    @staticmethod
    def _matches(chunk,filters):
        return not filters or all(v is None or str(chunk.get(k) or "").lower()==str(v).lower() for k,v in filters.items())
    def search(self,question:str,top_k:int=5,filters:dict|None=None,mode:str="hybrid") -> list[dict]:
        if not question.strip(): return []
        cache_key=(question.strip().lower(),top_k,tuple(sorted((filters or {}).items())),mode)
        if cache_key in self._cache:
            return [dict(x) for x in self._cache[cache_key]]
        allowed=[i for i,c in enumerate(self.chunks) if self._matches(c,filters)]
        if not allowed: return []
        kw=self.bm25.get_scores(tokenize(question))
        qv=self.model.encode([question],normalize_embeddings=True,convert_to_numpy=True)[0]
        vs=self.vectors@qv
        ko=[i for i in sorted(allowed,key=lambda i:kw[i],reverse=True) if kw[i]>0][:TOP_K_KEYWORD]
        vo=sorted(allowed,key=lambda i:vs[i],reverse=True)[:TOP_K_VECTOR]
        if mode=="keyword": ranked=[(i,float(kw[i])) for i in ko]
        elif mode=="vector": ranked=[(i,float(vs[i])) for i in vo]
        elif mode=="hybrid":
            fused={}
            for order in (ko,vo):
                for rank,idx in enumerate(order,1): fused[idx]=fused.get(idx,0)+1/(RRF_K+rank)
            ranked=sorted(fused.items(),key=lambda x:x[1],reverse=True)
        else: raise ValueError("mode must be keyword, vector, or hybrid")
        candidates=[i for i,_ in ranked[:max(top_k*3,10)]]
        if mode=="hybrid" and self.reranker and candidates:
            scores=self.reranker.predict([(question,self.chunks[i]["text"]) for i in candidates])
            ranked2=sorted(zip(candidates,scores),key=lambda x:float(x[1]),reverse=True)
            result=[(i,float(s),"hybrid+cross-encoder") for i,s in ranked2[:top_k]]
        else: result=[(i,float(s),mode) for i,s in ranked[:top_k]]
        out=[]
        for i,score,method in result:
            item=dict(self.chunks[i]); item.update(retrieval_score=score,retrieval_method=method); out.append(item)
        if len(self._cache) >= self._cache_limit:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key]=[dict(x) for x in out]
        return out
