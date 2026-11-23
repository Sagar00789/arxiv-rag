# arxiv-rag

**Ask questions about AI research papers and get cited answers, with every retrieval change measured.**

[![CI](https://github.com/Sagar00789/arxiv-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Sagar00789/arxiv-rag/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sagar00789/arxiv-rag/blob/main/notebooks/colab_quickstart.ipynb)

A retrieval-augmented generation (RAG) system over arXiv papers on RAG and LLM agents.
The focus is less on "it works" and more on **how well it works and why**: each improvement
(chunking, hybrid search, reranking) is scored on a fixed test set, and the results are
logged in [`eval/RESULTS.md`](eval/RESULTS.md).

## Architecture

```mermaid
flowchart LR
    A[arXiv API] --> B[PDF → text<br/>PyMuPDF]
    B --> C[Chunking<br/>300 words, 50 overlap]
    C --> D[Embeddings<br/>bge-small-en-v1.5]
    D --> E[(Qdrant)]
    Q[Question] --> R[Retriever]
    E --> R
    R --> G[LLM via LiteLLM<br/>cited answer]
    R --> V[Eval harness<br/>hit@k · MRR · latency]
```

## Results

Two test sets with the **same 50 gold passages**, asked two ways:
- **Original:** LLM-written while looking at the passage, so the questions reuse its words
  (keyword-friendly, the "easy" end).
- **Paraphrased:** rewritten as someone who hasn't read the paper would ask: synonyms, and
  methods described instead of named. Only 11% of its words appear in the original questions
  (the "hard" end). Real users sit somewhere in between.

**Chunk hit@5:** is the exact gold passage in the top 5?

| Retriever | Original | Paraphrased | Average | p50 latency (T4) |
|---|---|---|---|---|
| Dense (bge-small embeddings) | 0.60 | 0.22 | 0.41 | 17 ms |
| BM25 (keywords) | **0.88** | 0.26 | 0.57 | **5 ms** |
| Hybrid, equal weights (RRF) | 0.72 | **0.50** | 0.61 | 27 ms |
| **Hybrid, 80% BM25** (`hybrid@0.8`) | 0.82 | 0.46 | **0.64** | 28 ms |
| BM25 → cross-encoder rerank | **0.88** | 0.44 | **0.66** | 818 ms |
| Hybrid equal → rerank | 0.84 | _not run yet_ | | 878 ms |

Full metrics (MRR, paper-level hit@5) in [`eval/RESULTS.md`](eval/RESULTS.md). Corpus: 49 arXiv
papers on RAG and LLM agents → 1,681 chunks. With n = 50, the standard error of each hit rate is
about ±0.07, so gaps under ~0.1 aren't meaningful; gaps like 0.26 → 0.50 are.

### What the numbers say

1. **The "best" retriever depends on how people ask.** BM25 wins when the question shares the
   passage's words (0.88) and collapses when it doesn't (0.26). Dense retrieval is weak on both
   with this small embedding model.
2. **Hybrid search pays off exactly when its parts make different mistakes.** On original
   questions BM25 is right almost everywhere dense is, so equal fusion only adds dense's noise
   (0.72 < 0.88). On paraphrased questions they fail on *different* questions, and fusion nearly
   doubles either one alone (0.50 vs 0.26 / 0.22).
3. **`hybrid@0.8` is the best fast all-rounder:** close to BM25 on easy questions, close to
   equal hybrid on hard ones, with a 28 ms median latency. The weight was fixed *before* seeing
   these results, not tuned on them. **It's now the default retriever.**
4. **The reranker rescues keyword search on hard questions (0.26 → 0.44)** because it reads
   meaning, but it can only reorder the candidates it's given: if BM25 never retrieved the gold
   passage, reranking can't find it. And it costs about 30× the latency.
5. **A subtle one: `hybrid@0.8-rerank` scored exactly the same as `bm25-rerank`.** At 80/20,
   a passage found only by dense scores at most 0.2/61 ≈ 0.0033, while BM25's 30th result
   scores 0.8/90 ≈ 0.0089. So the reranker's 30 candidates are always exactly BM25's: the
   weighting silently removed dense from the candidate pool.

### Next experiments
- **Equal hybrid → rerank on paraphrased questions:** candidates from *both* retrievers should
  beat 0.44, since the reranker would finally see dense's finds.
- **A bigger embedding model** (bge-base / bge-large): dense is the weak link on both sets.
- **Answer-quality evals:** faithfulness and citation accuracy with an LLM judge.

## Quickstart

**Fastest:** click *Open in Colab* above and run all cells on a free T4 GPU.

**Locally:**
```bash
pip install -e ".[all,dev]"
cp .env.example .env              # add a free Gemini or Groq key
arxiv-rag ingest --max-results 50 # download and chunk papers
arxiv-rag index                   # embed into Qdrant
arxiv-rag ask "How does Self-RAG decide when to retrieve?"
arxiv-rag make-testset --n 50     # then review eval/testset.jsonl
arxiv-rag eval -r dense -r bm25 -r hybrid -r hybrid-rerank
arxiv-rag paraphrase-testset      # questions reworded away from the passage's wording
arxiv-rag eval -r bm25 -r hybrid@0.8 -r bm25-rerank --testset eval/testset_paraphrased.jsonl
uvicorn arxiv_rag.api:app --reload  # POST /ask at http://localhost:8000/docs
```

## Design decisions

- **LiteLLM for generation:** swap Gemini, Groq, Claude or GPT with one env var and compare them.
- **Embedded Qdrant:** a real vector DB with no server to run; the same code works against Qdrant Cloud.
- **Retrieval is evaluated separately from generation:** if the right chunk isn't retrieved,
  no prompt can fix the answer, so retrieval is measured first.
- **Reciprocal rank fusion for hybrid search:** it merges rankings by position, not score, so
  BM25 and cosine scores (on totally different scales) never need tuning against each other.
- **Rerank only a shortlist:** the cross-encoder is far more accurate but about 100× slower per
  passage than an embedding lookup, so it re-scores just the top 30 hybrid candidates.
- **Offline tests:** a hashing embedder and in-memory Qdrant let CI run in seconds with no keys.

## Project layout

```
src/arxiv_rag/
  ingest.py      arXiv search, PDF download, text extraction
  chunking.py    cleaning (drops reference lists) and overlapping windows
  embed.py       embedding interface + sentence-transformers
  store.py       Qdrant wrapper
  retrieve.py    dense, BM25, hybrid (RRF) and reranking retrievers
  generate.py    cited-answer prompt, LLM call
  evaluation.py  test-set generation, hit@k / MRR, results log
  pipeline.py    retrieval + generation
  api.py         FastAPI service
  cli.py         `arxiv-rag` command
tests/           offline unit tests
notebooks/       Colab GPU quickstart
```

## Roadmap

- [x] Baseline pipeline, CLI, API, eval harness, CI
- [x] Baseline numbers on a reviewed 50-question test set (49 papers, 1,681 chunks)
- [x] Hybrid retrieval (BM25 + dense with reciprocal rank fusion)
- [x] Cross-encoder reranking
- [x] Measure all retrievers on the test set (BM25 wins; see analysis above)
- [x] Paraphrased test set + word-overlap measurement (`paraphrase-testset`)
- [x] Weighted fusion (`hybrid@0.8`) and BM25 + rerank (`bm25-rerank`)
- [x] Results on both test sets → `hybrid@0.8` is the default retriever
- [ ] Answer quality evals (faithfulness and citation accuracy, LLM-as-judge)
- [ ] Tracing and cost tracking (Langfuse)
- [ ] Web UI + live deployment
