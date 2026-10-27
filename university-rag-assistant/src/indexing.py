import json, re
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import INDEX_DIR, EMBEDDING_MODEL

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",text.lower())

def build_index(chunks: list[dict], index_dir: Path=INDEX_DIR, model_name: str=EMBEDDING_MODEL):
    if not chunks: raise ValueError("No chunks found. Add documents under data/raw.")
    index_dir.mkdir(parents=True,exist_ok=True)
    model=SentenceTransformer(model_name)
    vectors=model.encode([c["text"] for c in chunks],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype("float32")
    np.save(index_dir/"vectors.npy",vectors)
    (index_dir/"chunks.json").write_text(json.dumps(chunks,ensure_ascii=False,indent=2),encoding="utf-8")
    (index_dir/"manifest.json").write_text(json.dumps({"embedding_model":model_name,"chunk_count":len(chunks),"embedding_dimension":int(vectors.shape[1]),"metric":"cosine via normalized dot product"},indent=2),encoding="utf-8")
    # BM25 is built from chunk text at startup; no unsafe Python pickle is required.
