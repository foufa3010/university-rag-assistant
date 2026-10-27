from src.ingest import load_documents,chunk_documents
from src.indexing import build_index
from src.config import CHUNK_WORDS,CHUNK_OVERLAP

def main():
    docs=load_documents()
    if not docs: raise SystemExit("No .txt/.md documents found in data/raw")
    chunks=chunk_documents(docs,CHUNK_WORDS,CHUNK_OVERLAP)
    build_index(chunks)
    print(f"Indexed {len(docs)} documents into {len(chunks)} chunks.")
if __name__=="__main__": main()
