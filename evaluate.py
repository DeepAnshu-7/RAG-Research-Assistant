# only checks the search part, no llm calls
# a chunk counts as correct if it's from the right paper and has one of the keywords
from rag import hybrid_search, keyword_search, load_index, rerank, semantic_search

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
    {"question": "What syndrome describes how chronic epilepsy damages the structure and function of the heart?", "paper": "paper8.pdf", "keywords": ["Epileptic Heart"]},
    {"question": "Which heart rhythm changes can concurrent ECG and EEG pick up during seizures?", "paper": "paper8.pdf", "keywords": ["tachycardi"]},
    {"question": "What does the epilepsy review recommend for ECG and EEG recording in epilepsy monitoring units?", "paper": "paper8.pdf", "keywords": ["standardized protocols"]},
    {"question": "How many people took part in the ECG and EEG stress experiment?", "paper": "paper9.pdf", "keywords": ["Eight subjects", "8 subjects"]},
    {"question": "What tasks did participants do while wearing the ECG and EEG sensors?", "paper": "paper9.pdf", "keywords": ["jog", "thrilling video"]},
    {"question": "Which brain waves increased together with heart rate when the stimulus was activated?", "paper": "paper9.pdf", "keywords": ["alpha, beta, delta", "theta"]},
    {"question": "How many MESA participants were followed in the deep survival network study?", "paper": "paper10.pdf", "keywords": ["6814"]},
    {"question": "Which deep learning survival model improved ASCVD risk prediction over the Cox model?", "paper": "paper10.pdf", "keywords": ["DeepSurv"]},
    {"question": "Did adding inflammatory or imaging biomarkers improve the cardiovascular risk models?", "paper": "paper10.pdf", "keywords": ["minimal/no significant", "minimal"]},
    {"question": "Which algorithm was used to predict all-cause mortality in obstructive sleep apnea patients?", "paper": "paper11.pdf", "keywords": ["light gradient-boosting", "LGBM"]},
    {"question": "What AUC did the model reach for predicting 10- and 15-year mortality in sleep apnea patients?", "paper": "paper11.pdf", "keywords": ["0.806"]},
    {"question": "Was parasympathetic activity higher or lower in sleep apnea patients with worse outcomes?", "paper": "paper11.pdf", "keywords": ["parasympathetic activity was higher"]},
    {"question": "How many participants came from MGH and SHHS in the study predicting health outcomes from sleep EEG?", "paper": "paper12.pdf", "keywords": ["8673", "5650"]},
    {"question": "For which outcome was the 10-year risk ratio between poor and good sleep not significant?", "paper": "paper12.pdf", "keywords": ["except for atrial"]},
    {"question": "How was death handled in the survival model for sleep EEG predicted outcomes?", "paper": "paper12.pdf", "keywords": ["competing risk"]},
    {"question": "How many single-lead ECG recordings were analyzed to predict atrial fibrillation from sleep studies?", "paper": "paper13.pdf", "keywords": ["18,782"]},
    {"question": "What signal quality cutoff was used to drop noisy ECG windows in the atrial fibrillation study?", "paper": "paper13.pdf", "keywords": ["SQI<0.95", "SQI < 0.95"]},
    {"question": "Which challenge data was used for transfer learning in the atrial fibrillation model?", "paper": "paper13.pdf", "keywords": ["PhysioNet Challenge 2021"]},
    {"question": "What overall accuracy and kappa did ECG-SleepNet achieve?", "paper": "paper14.pdf", "keywords": ["80.79%"]},
    {"question": "Which type of network classifies the five sleep stages in the last stage of ECG-SleepNet?", "paper": "paper14.pdf", "keywords": ["Kolmogorov-Arnold"]},
    {"question": "How accurate was ECG-SleepNet on the N1 sleep stage?", "paper": "paper14.pdf", "keywords": ["60.36"]},
    {"question": "How many patients' sleep recordings were used for ECG-based biological age estimation?", "paper": "paper15.pdf", "keywords": ["1149"]},
    {"question": "What hazard ratio for cardiovascular risk was linked to the predicted biological age?", "paper": "paper15.pdf", "keywords": ["3.56"]},
    {"question": "How were heart rhythm patterns grouped before estimating biological age?", "paper": "paper15.pdf", "keywords": ["unsupervised clustering", "time-series clusters"]},
    # same facts as some questions above, asked in everyday words like a real user would
    {"question": "How well did the sleep staging transformer agree with human scorers?", "paper": "paper4.pdf", "keywords": ["kappa"]},
    {"question": "How big was the ECG training data for the blood pressure model?", "paper": "paper3.pdf", "keywords": ["752,415", "750,000"]},
    {"question": "Which explainability technique did the coma prognosis study use?", "paper": "paper6.pdf", "keywords": ["SHAP"]},
    {"question": "Where did the data on unconscious patients after the heart stopped come from?", "paper": "paper6.pdf", "keywords": ["I-CARE"]},
    {"question": "What kind of network compressed heart signals into a compact representation?", "paper": "paper5.pdf", "keywords": ["autoencoder"]},
    {"question": "How much did the chance of dying rise for people with a fragmented sleep pattern?", "paper": "paper7.pdf", "keywords": ["33%"]},
    {"question": "Which public sleep cohort was used to study brain and heart ageing for stroke?", "paper": "paper2.pdf", "keywords": ["Sleep Heart Health Study", "SHHS"]},
    {"question": "How accurately could the AI spot high blood pressure from heart tracings?", "paper": "paper3.pdf", "keywords": ["0.803", "0.771"]},
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
    corpus, faiss_index, bm25_index = load_index()
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
