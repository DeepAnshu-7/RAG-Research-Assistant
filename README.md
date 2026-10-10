# RAG Research Assistant

Ask questions about research papers and get answers that come only from the papers, with the paper name and page number as the source.

Built on 7 papers about ECG, EEG and sleep studies.

## How it works

1. Read the PDFs page by page and split the text into small chunks.
2. Search the chunks two ways: by keywords (BM25) and by meaning (embeddings).
3. Combine both results and rerank them to find the best 4 chunks.
4. Send those chunks to an LLM, which writes the answer and cites its sources.

## Results

Tested on 16 questions with known answers:

| Search method | Correct chunk in top 3 |
|---|---|
| Meaning only (embeddings) | 56% |
| Keywords only (BM25) | 75% |
| Both + reranking (final) | **88%** |

Answers checked by a second LLM: 100% of answers were supported by the papers (Groq, gpt-oss-120b).

## Tech stack

Python, FAISS, BM25, Sentence-Transformers, Groq / Ollama, Streamlit

## Run it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key      # or install Ollama and run: ollama pull llama3.2:3b
streamlit run app.py
```

Other scripts:

```bash
python rag.py                # chat in the terminal
python evaluate.py           # test search quality
python evaluate_answers.py   # test answer quality (needs GROQ_API_KEY)
python test_rag.py           # unit tests
```

## Papers

The PDFs are not included because of copyright. Download them and save them in a `data/` folder as `paper1.pdf` to `paper7.pdf`:

1. [Multimodal cardiovascular risk profiling using self-supervised learning of polysomnography](https://doi.org/10.1093/sleep/zsaf371)
2. [Brain-Heart Aging During Sleep Predicts Incident Stroke](https://doi.org/10.1109/BHI67747.2025.11269528)
3. [A deep learning digital biomarker to detect hypertension and stratify cardiovascular risk from the electrocardiogram](https://doi.org/10.1038/s41746-025-01491-8)
4. [A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages](https://doi.org/10.1093/sleep/zsaf061)
5. [Unsupervised deep learning of electrocardiograms enables scalable human disease profiling](https://doi.org/10.1038/s41746-024-01418-9)
6. [Explainable machine learning model based on EEG, ECG, and clinical features for predicting neurological outcomes in cardiac arrest patient](https://doi.org/10.1038/s41598-025-93579-0)
7. [Continuous sleep depth index annotation with deep learning yields novel digital biomarkers for sleep health](https://doi.org/10.1038/s41746-025-01607-0)

You can also use your own PDFs. Only the evaluation scripts depend on these seven.
