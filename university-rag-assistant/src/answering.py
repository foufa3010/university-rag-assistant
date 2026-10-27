import requests
from src.config import OLLAMA_URL,OLLAMA_MODEL
ABSTENTION="I don't know based on the available university documents."

def rewrite_question(question:str)->str:
    # Deterministic baseline normalizer; preserves course codes and policy wording.
    return " ".join(question.strip().split())

def _generate(question,contexts):
    evidence="\n\n".join(f'[{c["chunk_id"]}] Source: {c["source_doc"]}; section: {c.get("section")}\n{c["text"]}' for c in contexts)
    prompt=f"""You are a university policy assistant. Answer using ONLY the evidence below. Do not use outside knowledge. If evidence does not support an answer, say exactly: {ABSTENTION} Cite every factual claim with a chunk ID in square brackets. If evidence conflicts or is incomplete, say so. Do not make an official graduation decision.

EVIDENCE:
{evidence}

QUESTION: {question}

Concise answer with citations:"""
    try:
        r=requests.post(f"{OLLAMA_URL.rstrip('/')}/api/generate",json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False},timeout=90)
        r.raise_for_status(); return r.json().get("response","").strip() or None
    except (requests.RequestException,ValueError): return None

def answer_question(question:str,contexts:list[dict],use_llm:bool=True)->dict:
    normalized=rewrite_question(question)
    if not contexts: return {"answer":ABSTENTION,"citations":[],"abstained":True}
    answer=_generate(normalized,contexts) if use_llm else None
    if not answer:
        # Conservative fallback: quote the top passage rather than fabricate a synthesis.
        c=contexts[0]; answer=f'{c["text"]} [{c["chunk_id"]}]'
    citations=[{"chunk_id":c["chunk_id"],"source_doc":c["source_doc"],"title":c["title"],"text":c["text"]} for c in contexts]
    return {"answer":answer,"citations":citations,"abstained":answer.strip()==ABSTENTION}
