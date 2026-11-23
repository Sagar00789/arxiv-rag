"""Retrieval evaluation: test-set generation, metrics and a results log.

Every experiment (chunk size, hybrid search, reranker, ...) is scored on the same test set
and appended to eval/RESULTS.md, so the README always shows measured gains.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk
from .retrieve import Retriever


@dataclass
class TestItem:
    __test__ = False  # not a pytest test class

    question: str
    gold_paper_id: str
    gold_chunk_id: str | None = None
    reference_answer: str | None = None


def load_testset(path: Path) -> list[TestItem]:
    with path.open(encoding="utf-8") as f:
        return [TestItem(**json.loads(line)) for line in f if line.strip()]


def save_testset(items: list[TestItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


# ---------- metrics ----------


def reciprocal_rank(ranked: list[str], gold: str) -> float:
    for i, x in enumerate(ranked, start=1):
        if x == gold:
            return 1.0 / i
    return 0.0


def hit_at_k(ranked: list[str], gold: str, k: int) -> float:
    return 1.0 if gold in ranked[:k] else 0.0


def evaluate_retrieval(retriever: Retriever, items: list[TestItem], k: int = 5) -> dict:
    chunk_hits, chunk_rr, paper_hits, paper_rr, latencies = [], [], [], [], []
    for it in items:
        t0 = time.perf_counter()
        hits = retriever.retrieve(it.question, k)
        latencies.append((time.perf_counter() - t0) * 1000)
        paper_ids = [h.chunk.paper_id for h in hits]
        paper_hits.append(hit_at_k(paper_ids, it.gold_paper_id, k))
        paper_rr.append(reciprocal_rank(paper_ids, it.gold_paper_id))
        if it.gold_chunk_id:
            chunk_ids = [h.chunk.chunk_id for h in hits]
            chunk_hits.append(hit_at_k(chunk_ids, it.gold_chunk_id, k))
            chunk_rr.append(reciprocal_rank(chunk_ids, it.gold_chunk_id))

    def mean(xs: list[float]) -> float | None:
        return round(statistics.fmean(xs), 4) if xs else None

    return {
        "n": len(items),
        "k": k,
        f"chunk_hit@{k}": mean(chunk_hits),
        "chunk_mrr": mean(chunk_rr),
        f"paper_hit@{k}": mean(paper_hits),
        "paper_mrr": mean(paper_rr),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
    }


# ---------- synthetic test set ----------

QUESTION_PROMPT = """Below is a passage from the research paper "{title}".

Write ONE specific question that this passage answers, the way a researcher would ask it \
without seeing the passage. Don't say "this passage" or "this paper"; name the method or \
concept instead. Then give a short answer taken from the passage.

Reply with JSON only: {{"question": "...", "answer": "..."}}

Passage:
{text}"""


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def make_testset(
    chunks: list[Chunk], n: int, model: str, seed: int = 42, min_words: int = 120
) -> list[TestItem]:
    """Sample chunks and have an LLM write one question per chunk (that chunk is the gold)."""
    from .generate import complete

    rng = random.Random(seed)
    candidates = [c for c in chunks if len(c.text.split()) >= min_words]
    rng.shuffle(candidates)
    items: list[TestItem] = []
    for c in candidates:
        if len(items) >= n:
            break
        prompt = QUESTION_PROMPT.format(title=c.title, text=c.text)
        data = _parse_json(complete([{"role": "user", "content": prompt}], model=model))
        if not data or not data.get("question"):
            continue
        items.append(
            TestItem(
                question=data["question"].strip(),
                gold_paper_id=c.paper_id,
                gold_chunk_id=c.chunk_id,
                reference_answer=(data.get("answer") or "").strip() or None,
            )
        )
    return items


# ---------- results log ----------


def record_results(results: dict, name: str, eval_dir: Path) -> Path:
    """Save full results as JSON and append one row to eval/RESULTS.md."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    (eval_dir / "results").mkdir(parents=True, exist_ok=True)
    out = eval_dir / "results" / f"{stamp.replace(' ', '_').replace(':', '')}_{name}.json"
    out.write_text(json.dumps({"name": name, "time": stamp, **results}, indent=2))

    k = results["k"]
    md = eval_dir / "RESULTS.md"
    if not md.exists():
        md.write_text(
            "# Retrieval results\n\n"
            "Every row is one run of `arxiv-rag eval` on the same test set.\n\n"
            f"| date (UTC) | experiment | n | chunk hit@{k} | chunk MRR | paper hit@{k} "
            "| paper MRR | p50 ms |\n|---|---|---|---|---|---|---|---|\n"
        )
    row = (
        f"| {stamp} | {name} | {results['n']} | {results.get(f'chunk_hit@{k}')} "
        f"| {results.get('chunk_mrr')} | {results.get(f'paper_hit@{k}')} "
        f"| {results.get('paper_mrr')} | {results.get('p50_latency_ms')} |\n"
    )
    with md.open("a", encoding="utf-8") as f:
        f.write(row)
    return out
