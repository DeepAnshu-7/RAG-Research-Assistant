import re

import streamlit as st

from rag import load_index, run_rag_pipeline

st.set_page_config(page_title="Research Paper Assistant")
st.title("Research Paper Assistant", anchor=False)

# runs only once, not on every rerun of the page
@st.cache_resource
def load_system():
    return load_index()


try:
    with st.spinner("Loading papers..."):
        corpus, faiss_index, bm25_index = load_system()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

# sort by the number in the name, so paper2 comes before paper10
papers = sorted(set((c["source"], c["title"]) for c in corpus),
                key=lambda p: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", p[0])])
st.write(
    f"Search across {len(papers)} research papers on ECG, EEG and sleep studies. "
    "Answers are generated only from the papers, with the source paper and page number for every claim."
)

with st.sidebar:
    st.subheader("Indexed papers", anchor=False)
    for source, title in papers:
        st.write(f"**{source}**: {title}")
    st.caption('Mention a paper by name (e.g. "paper3") to search only that paper.')

query = st.text_input("Question", placeholder="e.g. Which dataset was used to predict stroke risk?")

if query:
    answer, sources = run_rag_pipeline(query, corpus, faiss_index, bm25_index)

    st.subheader("Answer", anchor=False)
    st.write_stream(answer)

    st.subheader("Sources", anchor=False)
    for i, c in enumerate(sources, start=1):
        with st.expander(f"[{i}] {c['source']}: {c['title']}, page {c['page']}"):
            st.write(c["text"])
