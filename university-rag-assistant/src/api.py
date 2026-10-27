from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from src.retrieval import HybridRetriever
from src.answering import answer_question
app=FastAPI(title="University RAG Assistant API",version="1.0.0")
retriever=None
class AskRequest(BaseModel):
    question:str=Field(min_length=1,max_length=2000)
    top_k:int=Field(default=5,ge=1,le=10)
    mode:str=Field(default="hybrid",pattern="^(hybrid|keyword|vector)$")
    use_llm:bool=True
@app.on_event("startup")
def startup():
    global retriever
    try: retriever=HybridRetriever()
    except Exception as exc: raise RuntimeError("Index missing; run python -m src.build_index") from exc
@app.get("/health")
def health(): return {"status":"ok","index_loaded":retriever is not None}
@app.post("/ask")
def ask(req:AskRequest):
    if retriever is None: raise HTTPException(503,"Index not loaded")
    passages=retriever.search(req.question,req.top_k,mode=req.mode)
    return {"question":req.question,**answer_question(req.question,passages,req.use_llm),"retrieved_count":len(passages)}
