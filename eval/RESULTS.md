# Retrieval results

Every row is one run of `arxiv-rag eval` on the same test set.

| date (UTC) | experiment | n | chunk hit@5 | chunk MRR | paper hit@5 | paper MRR | p50 ms |
|---|---|---|---|---|---|---|---|
| 2026-11-23 18:44 | baseline-dense | 50 | 0.6 | 0.422 | 0.94 | 0.8807 | 16.6 |
| 2026-11-23 18:54 | dense | 50 | 0.6 | 0.422 | 0.94 | 0.8807 | 16.6 |
| 2026-11-23 18:54 | bm25 | 50 | 0.88 | 0.7183 | 0.98 | 0.9433 | 5.2 |
| 2026-11-23 18:54 | hybrid | 50 | 0.72 | 0.5883 | 0.96 | 0.94 | 26.7 |
| 2026-11-23 18:56 | hybrid-rerank | 50 | 0.84 | 0.7273 | 0.98 | 0.9367 | 877.6 |
