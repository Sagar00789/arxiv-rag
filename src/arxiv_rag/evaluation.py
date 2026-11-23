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
    chunks: list[Chunk],
    n: int,
    model: str,
    out_path: Path,
    seed: int = 42,
    min_words: int = 120,
) -> list[TestItem]:
    """Have an LLM write one question per sampled chunk (that chunk is the gold label).

    Each question is appended to `out_path` as soon as it exists, and a re-run resumes
    from what's already there, so a crash or quota limit never loses finished work.
    """
    import logging

    from .generate import complete

    log = logging.getLogger(__name__)

    items = load_testset(out_path) if out_path.exists() else []
    used = {it.gold_chunk_id for it in items}
    if items:
        log.info("Resuming: %d questions already in %s", len(items), out_path)

    rng = random.Random(seed)
    candidates = [c for c in chunks if len(c.text.split()) >= min_words]
    rng.shuffle(candidates)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for c in candidates:
            if len(items) >= n:
                break
            if c.chunk_id in used:
                continue
            prompt = QUESTION_PROMPT.format(title=c.title, text=c.text)
            data = _parse_json(complete([{"role": "user", "content": prompt}], model=model))
            if not data or not data.get("question"):
                continue
            item = TestItem(
                question=data["question"].strip(),
                gold_paper_id=c.paper_id,
                gold_chunk_id=c.chunk_id,
                reference_answer=(data.get("answer") or "").strip() or None,
            )
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            f.flush()
            items.append(item)
            used.add(c.chunk_id)
            if len(items) % 5 == 0:
                log.info("Generated %d/%d questions", len(items), n)
    return items


# ---------- paraphrased test set ----------

PARAPHRASE_PROMPT = """A researcher wants to find the passage below, but hasn't read it, so they \
won't use its wording. Rewrite the question the way they would ask it.

Rules:
- Same meaning; the passage must still answer it.
- Avoid the passage's distinctive words and phrases: use synonyms and plain descriptions.
- If the question names a specific method, model or dataset, describe what it is or does
  instead of naming it, unless the question becomes ambiguous without the name.
- One question, natural and specific.

Reply with JSON only: {{"question": "..."}}

Passage:
{text}

Original question: {question}"""


def paraphrase_testset(
    items: list[TestItem], chunks_by_id: dict[str, Chunk], model: str, out_path: Path
) -> list[TestItem]:
    """Rewrite each question so it no longer echoes the gold passage's wording.

    Gold labels are unchanged, so scores on the original and paraphrased sets are directly
    comparable. Resumable, like make_testset.
    """
    import logging

    from .generate import complete

    log = logging.getLogger(__name__)
    done = load_testset(out_path) if out_path.exists() else []
    finished = {it.gold_chunk_id for it in done}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for it in items:
            if it.gold_chunk_id in finished:
                continue
            chunk = chunks_by_id.get(it.gold_chunk_id or "")
            if chunk is None:
                continue
            prompt = PARAPHRASE_PROMPT.format(text=chunk.text, question=it.question)
            data = _parse_json(complete([{"role": "user", "content": prompt}], model=model))
            if not data or not data.get("question"):
                continue
            new = TestItem(
                question=data["question"].strip(),
                gold_paper_id=it.gold_paper_id,
                gold_chunk_id=it.gold_chunk_id,
                reference_answer=it.reference_answer,
            )
            f.write(json.dumps(asdict(new), ensure_ascii=False) + "\n")
            f.flush()
            done.append(new)
            finished.add(it.gold_chunk_id)
            if len(done) % 5 == 0:
                log.info("Paraphrased %d/%d questions", len(done), len(items))
    return done


def word_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that also appear in the passage.
    High overlap means keyword search has it easy."""
    from .retrieve import tokenize

    q = set(tokenize(question))
    return len(q & set(tokenize(passage))) / len(q) if q else 0.0


def mean_overlap(items: list[TestItem], chunks_by_id: dict[str, Chunk]) -> float | None:
    vals = [
        word_overlap(it.question, chunks_by_id[it.gold_chunk_id].text)
        for it in items
        if it.gold_chunk_id in chunks_by_id
    ]
    return round(statistics.fmean(vals), 3) if vals else None


# ---------- results log ----------


def record_results(results: dict, name: str, eval_dir: Path) -> Path:
    """Save full results as JSON and append one row to eval/RESULTS.md."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    (eval_dir / "results").mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.@-]+", "_", name).strip("_")
    out = eval_dir / "results" / f"{stamp.replace(' ', '_').replace(':', '')}_{safe}.json"
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
