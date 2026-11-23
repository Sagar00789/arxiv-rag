"""Answer-quality evaluation with an LLM judge.

Retrieval metrics say whether the right passage was *found*. These say whether the final
answer is any good:

- faithfulness: share of the answer's claims that are supported by the passages it was given
  (the opposite of hallucination)
- citation accuracy: share of [n] citations that point at a passage that really supports
  the sentence citing it
- correctness: does the answer agree with the reference answer? yes = 1, partial = 0.5, no = 0
- gold passage retrieved: was the passage the question was written from among the sources?
  Splits "retrieval failed" from "generation failed".

Every judged answer is appended to a JSONL log as soon as it's done, so runs are resumable
and every score can be inspected by hand.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import TestItem, _parse_json
from .generate import complete, format_context, generate_answer
from .retrieve import Hit, Retriever

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You are grading an answer produced by a question-answering system over \
research papers. The system was given ONLY the numbered passages below.

Passages:
{context}

Question: {question}

Reference answer (written from the source passage): {reference}

System answer:
{answer}

Grade strictly. Reply with JSON only:
{{
  "total_claims": <number of distinct factual claims in the system answer>,
  "supported_claims": <how many of them are directly supported by the passages>,
  "total_citations": <number of [n] citation markers in the system answer>,
  "correct_citations": <how many point to a passage that supports the sentence citing it>,
  "correctness": "yes" | "partial" | "no",   // agreement with the reference answer
  "refused": true | false,                   // did the system say it couldn't find the answer?
  "note": "<one short sentence on the main problem, or 'ok'>"
}}"""

_CORRECTNESS = {"yes": 1.0, "partial": 0.5, "no": 0.0}


@dataclass
class JudgedAnswer:
    question: str
    gold_chunk_id: str | None
    answer: str
    source_ids: list[str]
    gold_retrieved: bool
    refused: bool
    faithfulness: float | None
    citation_accuracy: float | None
    correctness: float | None
    note: str


def _ratio(num, den) -> float | None:
    try:
        num, den = int(num), int(den)
    except (TypeError, ValueError):
        return None
    return round(min(num, den) / den, 3) if den > 0 else None


def judge_answer(item: TestItem, answer: str, hits: list[Hit], judge_model: str) -> JudgedAnswer:
    prompt = JUDGE_PROMPT.format(
        context=format_context(hits),
        question=item.question,
        reference=item.reference_answer or "(none)",
        answer=answer,
    )
    data = _parse_json(complete([{"role": "user", "content": prompt}], model=judge_model)) or {}
    refused = bool(data.get("refused")) or "couldn't find this" in answer.lower()
    return JudgedAnswer(
        question=item.question,
        gold_chunk_id=item.gold_chunk_id,
        answer=answer,
        source_ids=[h.chunk.chunk_id for h in hits],
        gold_retrieved=item.gold_chunk_id in {h.chunk.chunk_id for h in hits},
        refused=refused,
        # A refusal makes no claims, so it can't be unfaithful: score faithfulness only
        # on real answers, and report the refusal rate separately.
        faithfulness=None
        if refused
        else _ratio(data.get("supported_claims"), data.get("total_claims")),
        citation_accuracy=_ratio(data.get("correct_citations"), data.get("total_citations")),
        correctness=_CORRECTNESS.get(str(data.get("correctness", "")).lower()),
        note=str(data.get("note", ""))[:300],
    )


def evaluate_answers(
    retriever: Retriever,
    items: list[TestItem],
    answer_model: str,
    judge_model: str,
    log_path: Path,
    k: int = 5,
) -> list[JudgedAnswer]:
    """Answer and judge each question, appending to `log_path`; resumes from it."""
    done: list[JudgedAnswer] = []
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            done = [JudgedAnswer(**json.loads(line)) for line in f if line.strip()]
    finished = {d.question for d in done}
    if done:
        log.info("Resuming: %d answers already judged in %s", len(done), log_path)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for it in items:
            if it.question in finished:
                continue
            hits = retriever.retrieve(it.question, k)
            answer = generate_answer(it.question, hits, answer_model)
            judged = judge_answer(it, answer, hits, judge_model)
            f.write(json.dumps(asdict(judged), ensure_ascii=False) + "\n")
            f.flush()
            done.append(judged)
            if len(done) % 5 == 0:
                log.info("Judged %d/%d answers", len(done), len(items))
    return done


def summarize(judged: list[JudgedAnswer]) -> dict:
    def mean(xs: list[float | None]) -> float | None:
        vals = [x for x in xs if x is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    answered = [j for j in judged if not j.refused]
    return {
        "n": len(judged),
        "faithfulness": mean([j.faithfulness for j in answered]),
        "citation_accuracy": mean([j.citation_accuracy for j in answered]),
        "correctness": mean([j.correctness for j in judged]),
        "refusal_rate": round(sum(j.refused for j in judged) / len(judged), 3) if judged else None,
        "gold_retrieved": mean([float(j.gold_retrieved) for j in judged]),
        # The diagnostic split: when the right passage WAS retrieved, how often is the answer
        # right? Low here = generation problem; low overall but high here = retrieval problem.
        "correctness_when_gold_retrieved": mean(
            [j.correctness for j in judged if j.gold_retrieved]
        ),
    }


def record_answer_results(summary: dict, name: str, eval_dir: Path) -> None:
    md = eval_dir / "ANSWERS.md"
    if not md.exists():
        md.write_text(
            "# Answer-quality results\n\n"
            "Each row: retrieve top 5 -> answer with the LLM -> grade with an LLM judge.\n"
            "Faithfulness and citation accuracy exclude refusals; correctness counts a refusal "
            "as wrong.\nPer-question details: `eval/answers/*.jsonl`.\n\n"
            "| date (UTC) | experiment | n | faithfulness | citation acc. | correctness "
            "| refusals | gold retrieved | correct when gold retrieved |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    s = summary
    row = (
        f"| {stamp} | {name} | {s['n']} | {s['faithfulness']} | {s['citation_accuracy']} "
        f"| {s['correctness']} | {s['refusal_rate']} | {s['gold_retrieved']} "
        f"| {s['correctness_when_gold_retrieved']} |\n"
    )
    with md.open("a", encoding="utf-8") as f:
        f.write(row)
