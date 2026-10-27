import pytest
from src.ingest import chunk_document
def test_chunking_metadata_and_content():
    doc={"doc_id":"test","title":"Test","source_doc":"test.txt","text":"First paragraph.\n\nSecond paragraph."}
    chunks=chunk_document(doc,chunk_words=10,overlap=2)
    assert chunks and chunks[0]["source_doc"]=="test.txt"
    assert "First" in " ".join(x["text"] for x in chunks)
def test_invalid_overlap():
    with pytest.raises(ValueError): chunk_document({"doc_id":"x","text":"words"},5,5)
