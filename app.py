import streamlit as st

from rag import build_corpus, build_keyword_index, build_semantic_index, run_rag_pipeline

st.set_page_config(page_title="Research Paper Assistant")
st.title("Research Paper Assistant", anchor=False)

st.write(
    "Search across 7 research papers on ECG, EEG and sleep studies. "
    "Answers are generated only from the papers, with the source paper and page number for every claim."

)

# runs only once, not on every rerun of the page
@st.cache_resource
def load_system():
    corpus = build_corpus()
    faiss_index = build_semantic_index(corpus)
    bm25_index = build_keyword_index(corpus)
    return corpus, faiss_index, bm25_index


with st.spinner("Loading papers..."):
    corpus, faiss_index, bm25_index = load_system()

with st.sidebar:
    st.subheader("Indexed papers", anchor=False)
    papers = sorted(set(c["source"] for c in corpus))
    for paper in papers:
        st.write(paper)
    st.caption('Mention a paper by name (e.g. "paper3") to search only that paper.')

query = st.text_input("Question", placeholder="e.g. Which dataset was used to predict stroke risk?")

if query:
    answer, sources = run_rag_pipeline(query, corpus, faiss_index, bm25_index)

    st.subheader("Answer", anchor=False)
    st.write_stream(answer)

    st.subheader("Sources", anchor=False)
    for i, c in enumerate(sources, start=1):
        with st.expander(f"[{i}] {c['source']}, page {c['page']}"):
            st.write(c["text"])
