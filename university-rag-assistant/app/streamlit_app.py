import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.retrieval import HybridRetriever
from src.answering import answer_question
st.set_page_config(page_title="University RAG Assistant",page_icon="🎓",layout="wide")
st.title("🎓 University RAG Assistant")
st.caption("Demo corpus only. Verify official academic decisions with Academic Affairs.")
@st.cache_resource
def load_retriever(): return HybridRetriever()
try: retriever=load_retriever()
except Exception as e:
    st.error(f"Index could not load: {e}"); st.info("Run `python -m src.build_index` from the repository root."); st.stop()
with st.sidebar:
    st.header("Settings"); mode=st.selectbox("Retrieval",["hybrid","keyword","vector"])
    k=st.slider("Passages",1,10,5); llm=st.checkbox("Use local Ollama model",True)
    source=st.selectbox("Source document filter",["All documents",*[c["source_doc"] for c in retriever.chunks if c.get("chunk_index")==0]])
question=st.text_input("Ask a question",placeholder="What are the prerequisites for CS301?")
if st.button("Ask",type="primary") and question.strip():
    filters=None if source=="All documents" else {"source_doc":source}
    with st.spinner("Retrieving evidence..."):
        passages=retriever.search(question,k,filters,mode); result=answer_question(question,passages,llm)
    st.subheader("Answer"); st.write(result["answer"])
    if result["abstained"]: st.warning("Insufficient evidence was found.")
    st.subheader("Citations and retrieved passages")
    for p in passages:
        with st.expander(f'{p["source_doc"]} · {p["chunk_id"]} · score {p["retrieval_score"]:.4f}'):
            st.write(p["text"]); st.caption(f'Method: {p["retrieval_method"]}; title: {p["title"]}')
