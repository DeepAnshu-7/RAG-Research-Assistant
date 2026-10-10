# Research Paper Assistant — Hybrid RAG

Ask questions about a folder of research PDFs (ECG / EEG / sleep studies) and get answers **grounded in the papers, with citations to the exact paper and page**. Runs fully local (Ollama) or on Groq.

```
                 ┌──────────── indexing (once) ─────────────┐
 PDFs ──► page text ──► drop References ──► 150-word chunks (30 overlap)
                                               │            │
                                     MiniLM embeddings    BM25 index
                                        (FAISS, cosine)
                 └──────────────────────────────────────────┘

 query ──► [paper filter?] ──► semantic rank ─┐
                               BM25 rank ─────┴─► RRF fusion ──► top 20
        ──► cross-encoder rerank ──► top 4 chunks ──► LLM (cite [n]) ──► answer + sources
```

## Results

Retrieval evaluated on 16 hand-written questions with known answer locations across 7 papers (`python evaluate.py`). A retrieved chunk counts as relevant if it is from the right paper and contains the answer keyword.

| Retriever                  | Hit@1 | Hit@3 | MRR@5 |
|----------------------------|------:|------:|------:|
| Semantic only (MiniLM)     | 38%   | 56%   | 0.50  |
| BM25 only                  | 50%   | 75%   | 0.62  |
| Hybrid (RRF)               | 38%   | 75%   | 0.56  |
| **Hybrid + rerank (final)**| **69%** | **88%** | **0.79** |

Answer quality, judged by a different LLM (`qwen/qwen3.8-27b` on Groq) on the same 16 questions (`python evaluate_answers.py`):

| Answer model | Faithfulness | Answered | Correct |
|---|---:|---:|---:|
| `openai/gpt-oss-120b` (Groq) | 100% | 100% | 100% |
| `llama3.2:3b` (local Ollama) | 94% | 100% | 100% |

*Faithfulness* = every claim is supported by the retrieved chunks. The one local failure mixed two facts from the same paper.

Other findings from tuning:
- Removing the **References** section raised Hit@3 of the final pipeline from 75% → 88%. Bibliographies contain other papers' titles, which pulled retrieval toward the wrong paper.
- 250-word chunks hurt semantic search: the embedding model truncates at 256 tokens, so the end of each chunk was never embedded.

## Latency (MacBook, local llama3.2:3b)

| Stage | Before | After | Fix |
|---|---:|---:|---|
| App startup | 25 s | 6 s | Load models from local cache (`local_files_only`); Hugging Face was checking online on every start |
| Retrieval (hybrid + rerank) | 0.3 s | 0.3 s | — |
| Time to first word | ~4 s (full answer at once) | ~2.3 s | Token streaming; `keep_alive` keeps the model in memory |

The remaining delay is LLM prefill (reading the ~1k-token prompt). With Groq it is under 1 s.

## Design choices

| Choice | Why |
|---|---|
| Hybrid BM25 + embeddings | Embeddings catch paraphrases; BM25 catches exact terms (dataset names, numbers like `752,415`, `HTN-AI`) that small embedding models blur. |
| Reciprocal Rank Fusion | BM25 scores (0–30) and cosine (0–1) aren't comparable. RRF uses ranks only, so no normalization or weight tuning is needed. |
| Cross-encoder reranker | The bi-encoder embeds query and chunk separately (fast, approximate). The cross-encoder reads them together (slow, accurate), so it only runs on the top 20. |
| Page-level chunks | Every chunk keeps `source` + `page` → verifiable citations. |
| Metadata filter | "what dataset does paper3 use" searches only `paper3.pdf`. |
| Grounded prompt | Answer only from context, cite `[n]`, otherwise say "I don't have enough information". |
| Ollama / Groq | Local by default (private, free). Set `GROQ_API_KEY` for a larger hosted model. |

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b          # or: export GROQ_API_KEY=...

streamlit run app.py              # web UI
python rag.py                     # CLI chat
python evaluate.py                # retrieval benchmark
python evaluate_answers.py        # answer quality, LLM judge (needs GROQ_API_KEY; --local for Ollama answers)
python test_rag.py                # unit tests
```

Swap the LLM with `OLLAMA_MODEL=qwen2.5:7b` or `GROQ_MODEL=...`.

## Papers

The PDFs are not included in this repo because of copyright. Download them from the links below and save them in a `data/` folder with these filenames (the evaluation questions refer to them by name). Any other PDFs work too; only the evaluation is tied to these seven.

| File | Paper |
|---|---|
| `paper1.pdf` | Multimodal cardiovascular risk profiling using self-supervised learning of polysomnography — [doi:10.1093/sleep/zsaf371](https://doi.org/10.1093/sleep/zsaf371) |
| `paper2.pdf` | Brain-Heart Aging During Sleep Predicts Incident Stroke — [doi:10.1109/BHI67747.2025.11269528](https://doi.org/10.1109/BHI67747.2025.11269528) |
| `paper3.pdf` | A deep learning digital biomarker to detect hypertension and stratify cardiovascular risk from the electrocardiogram — [doi:10.1038/s41746-025-01491-8](https://doi.org/10.1038/s41746-025-01491-8) |
| `paper4.pdf` | A foundational transformer leveraging full night, multichannel sleep study data accurately classifies sleep stages — [doi:10.1093/sleep/zsaf061](https://doi.org/10.1093/sleep/zsaf061) |
| `paper5.pdf` | Unsupervised deep learning of electrocardiograms enables scalable human disease profiling — [doi:10.1038/s41746-024-01418-9](https://doi.org/10.1038/s41746-024-01418-9) |
| `paper6.pdf` | Explainable machine learning model based on EEG, ECG, and clinical features for predicting neurological outcomes in cardiac arrest patient — [doi:10.1038/s41598-025-93579-0](https://doi.org/10.1038/s41598-025-93579-0) |
| `paper7.pdf` | Continuous sleep depth index annotation with deep learning yields novel digital biomarkers for sleep health — [doi:10.1038/s41746-025-01607-0](https://doi.org/10.1038/s41746-025-01607-0) |

## Limitations / next steps
- The eval set is small (16 questions), so treat the numbers as directional. An LLM judge can also be wrong. Next: a larger set with paraphrased questions.
- Indexes are rebuilt on startup (~6 s for 7 papers). Persist embeddings to disk for larger corpora.
- Flat FAISS index (brute force). Switch to HNSW/IVF at ~1M chunks.
- PDF text extraction loses tables and figures. Layout-aware parsing would help.
- No conversation memory: each question is independent.
