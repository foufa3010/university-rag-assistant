from pathlib import Path
import re
from src.config import RAW_DIR, CHUNK_WORDS, CHUNK_OVERLAP

def load_documents(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Read text/Markdown files; filename is a stable document ID."""
    docs=[]
    for path in sorted([*raw_dir.glob("*.txt"), *raw_dir.glob("*.md")]):
        text=path.read_text(encoding="utf-8").strip()
        if not text: continue
        # Optional front matter fields are retained if present; otherwise metadata is explicit null.
        metadata = {"title": path.stem.replace("_", " ").title(), "date": None, "author": None, "section": None}
        # Read simple metadata headers when supplied as `Title: ...`, `Date: ...`, etc.
        for line in text.splitlines()[:12]:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in metadata and value.strip():
                    metadata[key] = value.strip()
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if metadata["title"] == path.stem.replace("_", " ").title() and first_line and "—" in first_line:
            metadata["title"] = first_line.split("—", 1)[0].strip()
        docs.append({"doc_id":path.stem, "source_doc":path.name, **metadata, "text":text})
    return docs

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n",text) if p.strip()]

def chunk_document(doc: dict, chunk_words: int=CHUNK_WORDS, overlap: int=CHUNK_OVERLAP) -> list[dict]:
    """Paragraph-aware chunks; long paragraphs split into overlapping word windows."""
    if chunk_words < 1 or overlap < 0 or overlap >= chunk_words:
        raise ValueError("Require chunk_words > overlap >= 0.")
    segments=[]; current=[]
    for paragraph in split_paragraphs(doc["text"]):
        words=paragraph.split()
        if len(words)>chunk_words:
            if current: segments.append(" ".join(current)); current=[]
            start=0
            while start<len(words):
                segments.append(" ".join(words[start:start+chunk_words]))
                if start+chunk_words>=len(words): break
                start += chunk_words-overlap
        else:
            if current and len(current)+len(words)>chunk_words:
                segments.append(" ".join(current)); current=current[-overlap:] if overlap else []
            current.extend(words)
    if current: segments.append(" ".join(current))
    out=[]
    for i,txt in enumerate(segments):
        out.append({k:doc.get(k) for k in ("doc_id","title","source_doc","date","author","section")} | {
            "chunk_id":f'{doc["doc_id"]}::chunk_{i:04d}',"chunk_index":i,"text":txt})
    return out

def chunk_documents(docs, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    return [chunk for doc in docs for chunk in chunk_document(doc,chunk_words,overlap)]
