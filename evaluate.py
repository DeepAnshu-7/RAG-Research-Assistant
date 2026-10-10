# only checks the search part, no llm calls
# a chunk counts as correct if it's from the right paper and has one of the keywords
from rag import (build_corpus, build_keyword_index, build_semantic_index, hybrid_search,
                 keyword_search, rerank, semantic_search)

test_set = [
    {"question": "How many sleep studies was PFTSleep trained on and what kappa did it get?", "paper": "paper4.pdf", "keywords": ["13 888", "kappa"]},
    {"question": "Which foundation model classifies sleep stages from full-night multichannel data?", "paper": "paper4.pdf", "keywords": ["PFTSleep"]},
    {"question": "How many ECGs were used to train the hypertension detection model?", "paper": "paper3.pdf", "keywords": ["752,415", "750,000"]},
    {"question": "What AUROC did HTN-AI achieve for detecting hypertension?", "paper": "paper3.pdf", "keywords": ["0.803", "0.771"]},
    {"question": "Which regression was used to link predicted hypertension with cardiovascular outcomes?", "paper": "paper3.pdf", "keywords": ["Fine-Gray"]},
    {"question": "What kind of neural network was used to learn ECG encodings for disease profiling?", "paper": "paper5.pdf", "keywords": ["autoencoder"]},
    {"question": "How many prevalent and incident diseases were associated with the ECG latent space?", "paper": "paper5.pdf", "keywords": ["645", "606"]},
    {"question": "Which database of comatose cardiac arrest patients was analyzed?", "paper": "paper6.pdf", "keywords": ["I-CARE"]},
    {"question": "How was neurological outcome measured after cardiac arrest?", "paper": "paper6.pdf", "keywords": ["Cerebral Performance Category", "CPC"]},
    {"question": "Which method was used to explain feature contributions in the cardiac arrest model?", "paper": "paper6.pdf", "keywords": ["SHAP"]},
    {"question": "What is the continuous sleep depth index and how many recordings were used?", "paper": "paper7.pdf", "keywords": ["10,000", "sleep depth index"]},
    {"question": "How much higher was mortality risk in the disturbed sleep subtype?", "paper": "paper7.pdf", "keywords": ["33%"]},
    {"question": "Which dataset was used to predict incident stroke from brain-heart aging?", "paper": "paper2.pdf", "keywords": ["Sleep Heart Health Study", "SHHS"]},
    {"question": "Which sleep features gave the strongest predictive power for stroke risk?", "paper": "paper2.pdf", "keywords": ["deep sleep"]},
    {"question": "How many participants were used for training and external validation in the self-supervised polysomnography CVD study?", "paper": "paper1.pdf", "keywords": ["4398", "1093"]},
    {"question": "Which modality was predictive of incident hypertension in the polysomnography self-supervised study?", "paper": "paper1.pdf", "keywords": ["EEG-derived"]},
]


def is_relevant(chunk, paper, keywords):
    if chunk["source"] != paper:
        return False
    for kw in keywords:
        if kw.lower() in chunk["text"].lower():
            return True
    return False


def search(method, query, corpus, faiss_index, bm25_index, top_k=5):
    if method == "semantic":
        ids = semantic_search(query, faiss_index, corpus)
        return [corpus[i] for i in ids[:top_k]]
    if method == "bm25":
        ids = keyword_search(query, bm25_index)
        return [corpus[i] for i in ids[:top_k]]
    if method == "hybrid":
        return hybrid_search(query, corpus, faiss_index, bm25_index, top_k=top_k)
    if method == "hybrid + rerank":
        candidates = hybrid_search(query, corpus, faiss_index, bm25_index, top_k=20)
        return rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    corpus = build_corpus()
    faiss_index = build_semantic_index(corpus)
    bm25_index = build_keyword_index(corpus)
    print(f"{len(corpus)} chunks, {len(test_set)} test questions\n")

    print(f"{'method':<18}{'Hit@1':>8}{'Hit@3':>8}{'MRR@5':>8}")
    for method in ["bm25", "semantic", "hybrid", "hybrid + rerank"]:
        hit1 = 0
        hit3 = 0
        mrr = 0

        for item in test_set:
            results = search(method, item["question"], corpus, faiss_index, bm25_index)
            correct = [is_relevant(c, item["paper"], item["keywords"]) for c in results]

            if correct[0]:
                hit1 += 1
            if True in correct[:3]:
                hit3 += 1
            # MRR: 1/rank of the first correct chunk
            for rank, ok in enumerate(correct, start=1):
                if ok:
                    mrr += 1 / rank
                    break

        n = len(test_set)
        print(f"{method:<18}{hit1 / n:>8.0%}{hit3 / n:>8.0%}{mrr / n:>8.2f}")
