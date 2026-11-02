# RAG Research Assistant

Ask questions about research papers and get answers that come only from the papers, with the paper name and page number as the source.

Built on 15 papers about ECG, EEG and sleep studies.

## How it works

1. Read the PDFs page by page and split the text into small chunks.
2. Search the chunks two ways: by keywords (BM25) and by meaning (embeddings).
3. Combine both results and rerank them to find the best 4 chunks.
4. Send those chunks to an LLM, which writes the answer and cites its sources.

## Results

Tested on 48 questions with known answers across all 15 papers. A chunk counts as correct if it comes from the right paper and contains the expected fact (a keyword).

**Search quality**

| Search method | Right chunk is #1 | Right chunk in top 3 | MRR |
|---|---|---|---|
| Meaning only (embeddings) | 27% | 54% | 0.43 |
| Keywords only (BM25) | 44% | 71% | 0.57 |
| Both combined | 35% | 71% | 0.53 |
| Both + reranking (final) | **65%** | **81%** | **0.72** |

MRR shows how high the first correct chunk is ranked. 1.0 is the best score.

Reranking made the biggest difference. Combining the two searches without reranking was not better than keywords alone.

**Answer quality**

A second LLM (a different model from the one that writes the answers) read each answer next to the chunks it was given:

| Check | Result |
|---|---|
| Answer contains the expected fact | 83% |
| Answer is supported by the retrieved chunks | 98% |

Answers are written by Groq gpt-oss-120b.

**Weak spot:** questions written in everyday words ("high blood pressure" instead of "hypertension") find the right chunk in the top 3 only 3 out of 8 times.

**Limits of this test:** 48 questions is a small set, so one question moves a score by about 2 points. Matching by keyword favors exact words, and an LLM judge can make mistakes too.

## Tech stack

Python, FAISS, BM25, Sentence-Transformers, Groq / Ollama, Streamlit

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # or install Ollama and run: ollama pull llama3.2:3b
# streamlit run app.py
python -m streamlit run app.py
```

Other scripts:

```bash
python rag.py                # chat in the terminal
python evaluate.py           # test search quality
python evaluate_answers.py   # test answer quality (needs GROQ_API_KEY)
python test_rag.py           # unit tests
```

## Papers

The PDFs are not included because of copyright. Download them and save them in a `data/` folder as `paper1.pdf` to `paper15.pdf`:

1. [Multimodal cardiovascular risk profiling using self-supervised learning of polysomnography](https://doi.org/10.1093/sleep/zsaf371)
2. [Brain-Heart Aging During Sleep Predicts Incident Stroke](https://doi.org/10.1109/BHI67747.2025.11269528)
3. [A deep learning digital biomarker to detect hypertension and stratify cardiovascular risk from the electrocardiogram](https://doi.org/10.1038/s41746-025-01491-8)
4. [A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages](https://doi.org/10.1093/sleep/zsaf061)
5. [Unsupervised deep learning of electrocardiograms enables scalable human disease profiling](https://doi.org/10.1038/s41746-024-01418-9)
6. [Explainable machine learning model based on EEG, ECG, and clinical features for predicting neurological outcomes in cardiac arrest patient](https://doi.org/10.1038/s41598-025-93579-0)
7. [Continuous sleep depth index annotation with deep learning yields novel digital biomarkers for sleep health](https://doi.org/10.1038/s41746-025-01607-0)
8. [The brain-heart connection: Value of concurrent ECG and EEG recordings in epilepsy management](https://doi.org/10.1016/j.ebr.2024.100726)
9. [Cardiac and Brain Activity Correlation Analysis Using Electrocardiogram and Electroencephalogram Signals](https://scholar.google.com/scholar?q=Cardiac+and+Brain+Activity+Correlation+Analysis+Using+Electrocardiogram+and+Electroencephalogram+Signals) (IEEE HSI 2018)
10. [Deep neural survival networks for cardiovascular risk prediction: The Multi-Ethnic Study of Atherosclerosis (MESA)](https://doi.org/10.1016/j.compbiomed.2021.104983)
11. [Predicting All-Cause Mortality in Patients With Obstructive Sleep Apnea Using Sleep-Related Features: A Machine-Learning Approach](https://doi.org/10.3988/jcn.2024.0038)
12. [Assessing Risk of Health Outcomes From Brain Activity in Sleep](https://doi.org/10.1212/CPJ.0000000000200225)
13. [From Sleep Patterns to Heart Rhythms: Predicting Atrial Fibrillation from Overnight Polysomnograms](https://doi.org/10.1101/2024.06.04.24308444)
14. [ECG-SleepNet: Deep Learning-Based Comprehensive Sleep Stage Classification Using ECG Signals](https://arxiv.org/abs/2412.01929)
15. [Cardiovascular Risk Assessment via Sleep Patterns and ECG-Based Biological Age Estimation](https://doi.org/10.3390/jcm14103339)

You can also use your own PDFs. Only the evaluation scripts depend on these.

<img width="1111" height="783" alt="demo" src="https://github.com/user-attachments/assets/12c7f872-d412-4cbe-ac20-754251251b61" />