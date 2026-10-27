from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
EVAL_FILE = ROOT / "data" / "eval" / "questions.jsonl"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "35"))
TOP_K_KEYWORD = int(os.getenv("TOP_K_KEYWORD", "10"))
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "10"))
RRF_K = 60
