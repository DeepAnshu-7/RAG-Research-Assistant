import logging
import os
import pickle
import re

import faiss
import numpy as np
import ollama
from groq import Groq
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# some of the pdfs have broken objects, pypdf prints a lot of warnings for them
logging.getLogger("pypdf").setLevel(logging.ERROR)

# load from local cache first, otherwise huggingface checks online every time
# and startup was taking ~20 sec
try:
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
except OSError:
    # first run, models not downloaded yet
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_corpus(data_folder="data"):
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"'{data_folder}' folder not found. Create it and put the PDFs inside (see README)")

    corpus = []
    for filename in sorted(os.listdir(data_folder)):
        if not filename.endswith(".pdf"):
            continue

        try:
            reader = PdfReader(os.path.join(data_folder, filename))
        except Exception as e:
            # a broken pdf shouldn't stop the other papers from loading
            print(f"Skipping {filename}: {e}")
            continue
        # title saved inside the pdf, if there is none just use the filename
        title = (reader.metadata.title if reader.metadata else None) or filename
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # stop at the references section. it has titles of other papers
            # and search was returning those chunks for wrong questions
            # some papers number it, like "VII. REFERENCES"
            match = re.search(r"^\s*([ivx]+\.|\d+\.?)?\s*(references|bibliography)\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                text = text[:match.start()]

            for chunk in chunk_text(text):
                corpus.append({"text": chunk, "source": filename, "title": title, "page": page_num})

            if match:
                break

    if len(corpus) == 0:
        raise ValueError(f"No text found in PDFs inside '{data_folder}' folder")
    return corpus


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def build_semantic_index(corpus, embeddings=None):
    if embeddings is None:
        texts = [c["text"] for c in corpus]
        embeddings = embed_model.encode(texts, normalize_embeddings=True)

    # vectors are normalized, so inner product = cosine similarity
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index


def build_keyword_index(corpus):
    tokenized_corpus = [tokenize(c["text"]) for c in corpus]
    return BM25Okapi(tokenized_corpus)


def load_index(data_folder="data"):
    # reading the pdfs and making embeddings takes ~7 sec, so save them and reuse
    # until a pdf is added, removed or changed (or rag.py itself is edited)
    cache_file = os.path.join(data_folder, "index_cache.pkl")
    fingerprint = None
    if os.path.isdir(data_folder):
        pdfs = sorted(f for f in os.listdir(data_folder) if f.endswith(".pdf"))
        fingerprint = [(f, os.path.getsize(os.path.join(data_folder, f)), os.path.getmtime(os.path.join(data_folder, f))) for f in pdfs]
        fingerprint.append(os.path.getmtime(__file__))

    saved = None
    if fingerprint and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                saved = pickle.load(f)
        except Exception:
            saved = None  # broken cache file, just build again

    if saved and saved["fingerprint"] == fingerprint:
        corpus, embeddings = saved["corpus"], saved["embeddings"]
    else:
        corpus = build_corpus(data_folder)
        embeddings = embed_model.encode([c["text"] for c in corpus], normalize_embeddings=True)
        with open(cache_file, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "corpus": corpus, "embeddings": embeddings}, f)

    return corpus, build_semantic_index(corpus, embeddings), build_keyword_index(corpus)


def semantic_search(query, faiss_index, corpus):
    query_embedding = embed_model.encode([query], normalize_embeddings=True)
    # searching all chunks, ok for a few papers
    scores, indices = faiss_index.search(np.array(query_embedding).astype("float32"), len(corpus))
    return indices[0].tolist()


def keyword_search(query, bm25_index):
    scores = bm25_index.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1]
    return ranked.tolist()


def reciprocal_rank_fusion(rankings, k=60):
    # using rank instead of score, bm25 and cosine scores are on different scales
    scores = {}
    for ranking in rankings:
        for position, idx in enumerate(ranking):
            if idx not in scores:
                scores[idx] = 0
            scores[idx] += 1 / (k + position + 1)

    return sorted(scores, key=lambda idx: scores[idx], reverse=True)


def find_paper_in_query(query):
    # "what dataset is used in paper 3" -> paper3.pdf
    match = re.search(r"paper\s*(\d+)", query.lower())
    if match:
        return f"paper{match.group(1)}.pdf"
    return None


def hybrid_search(query, corpus, faiss_index, bm25_index, top_k=20, paper=None):
    semantic_ranking = semantic_search(query, faiss_index, corpus)
    keyword_ranking = keyword_search(query, bm25_index)
    combined = reciprocal_rank_fusion([semantic_ranking, keyword_ranking])

    if paper:
        combined = [idx for idx in combined if corpus[idx]["source"] == paper]

    return [corpus[idx] for idx in combined[:top_k]]


def rerank(query, candidates, top_k=4):
    if len(candidates) == 0:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, score in scored_candidates[:top_k]]


def generate_answer(query, chunks):
    # e.g. user asked about paper9 but there is no paper9.pdf
    if len(chunks) == 0:
        yield "I don't have enough information in these papers."
        return

    context = ""
    for i, c in enumerate(chunks, start=1):
        context += f"[{i}] ({c['source']}: {c['title']}, page {c['page']})\n{c['text']}\n\n"

    prompt = f"""You are a research assistant. Answer the question using ONLY the numbered context below.
Write the answer in 1-4 full sentences. End each sentence with the number of the context it came from, e.g. [1] or [2][3].
Refer to papers by their exact filename (e.g. paper2.pdf), never invent names.
If the context does not contain the answer, reply exactly: "I don't have enough information in these papers."

Context:
{context}
Question: {query}

Answer:"""

    messages = [{"role": "user", "content": prompt}]

    # stream it so text shows up while it's still generating
    if os.environ.get("GROQ_API_KEY"):
        client = Groq()
        response = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=messages,
            temperature=0,
            stream=True,
        )
        for part in response:
            yield part.choices[0].delta.content or ""
    else:
        response = ollama.chat(
            model=os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
            messages=messages,
            options={"temperature": 0},
            keep_alive="1h",  # ollama unloads the model after 5 min idle by default
            stream=True,
        )
        for part in response:
            yield part["message"]["content"]


def run_rag_pipeline(query, corpus, faiss_index, bm25_index):
    paper = find_paper_in_query(query)
    candidates = hybrid_search(query, corpus, faiss_index, bm25_index, top_k=20, paper=paper)
    top_chunks = rerank(query, candidates, top_k=4)
    answer = generate_answer(query, top_chunks)
    return answer, top_chunks


if __name__ == "__main__":
    print("Loading papers and building indexes...")
    corpus, faiss_index, bm25_index = load_index()
    print(f"Ready! {len(corpus)} chunks loaded. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if query == "":
            continue

        answer, sources = run_rag_pipeline(query, corpus, faiss_index, bm25_index)

        print("\nAssistant: ", end="")
        for word in answer:
            print(word, end="", flush=True)
        print("\n")

        for i, c in enumerate(sources, start=1):
            print(f"  [{i}] {c['source']} - {c['title']} - page {c['page']}")
        print()
