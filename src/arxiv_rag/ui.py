"""Framework-free helpers for the web demo (streamlit_app.py), kept here so they can be tested."""

from __future__ import annotations

import html
import time
from collections.abc import Callable
from pathlib import Path

from .pipeline import Answer, RAGPipeline

REPO_URL = "https://github.com/Sagar00789/arxiv-rag"

RETRIEVER_CHOICES = [
    ("Hybrid, 80% BM25 (default, best all-rounder)", "hybrid@0.8"),
    ("BM25 keywords only (fastest)", "bm25"),
    ("Dense embeddings only", "dense"),
    ("Hybrid, equal weights", "hybrid"),
]

EXAMPLES = [
    "How does Self-RAG decide when to retrieve?",
    "What are common ways to evaluate retrieval-augmented generation systems?",
    "How do multi-agent systems handle conflicting evidence from retrieved documents?",
    "What is reciprocal rank fusion?",
]

PipelineFactory = Callable[[str], RAGPipeline]


def score_label(retriever: str) -> str:
    """What the number next to each source means, which differs by retriever."""
    if retriever.endswith("-rerank"):
        return "rerank score"
    base = retriever.split("@")[0]
    return {
        "dense": "cosine similarity",
        "bm25": "BM25 score",
        "hybrid": "fusion score (RRF, ranks only, always small)",
    }.get(base, "score")


def format_sources(ans: Answer, retriever: str = "") -> str:
    if not ans.sources:
        return "_No passages retrieved._"
    label = score_label(retriever)
    parts = []
    for i, h in enumerate(ans.sources, start=1):
        c = h.chunk
        snippet = html.escape(c.text[:600] + ("…" if len(c.text) > 600 else ""))
        parts.append(
            f"**[{i}] [{html.escape(c.title)}](https://arxiv.org/abs/{c.paper_id})**  \n"
            f"<sub>arXiv:{c.paper_id} · passage {c.position} · {label} {h.score:.3f}</sub>\n"
            f"<details><summary>Show passage</summary>\n\n{snippet}\n\n</details>"
        )
    return "\n\n".join(parts)


def answer_question(question: str, retriever: str, factory: PipelineFactory) -> tuple[str, str]:
    """Returns (answer markdown, sources markdown). Errors become readable messages."""
    from .generate import LLMError
    from .pipeline import IndexMissingError

    question = (question or "").strip()
    if len(question) < 3:
        return "Please type a question.", ""
    t0 = time.perf_counter()
    try:
        ans = factory(retriever).ask(question)
    except LLMError as e:
        return f"⚠️ The language model is unavailable right now: {e}", ""
    except IndexMissingError as e:
        return f"⚠️ {e}", ""
    took = time.perf_counter() - t0
    footer = (
        f"\n\n<sub>{retriever} · {len(ans.sources)} passages · "
        f"retrieval {ans.retrieval_s * 1000:.0f} ms · LLM {ans.generation_s:.1f} s · "
        f"total {took:.1f} s</sub>"
    )
    return ans.answer + footer, format_sources(ans, retriever)


def results_markdown(readme: Path = Path("README.md")) -> str:
    """The results section of the README, so the demo shows the evaluation too."""
    try:
        text = readme.read_text(encoding="utf-8")
        start = text.index("## Results")
        end = text.index("## Quickstart", start)
        return text[start:end]
    except (OSError, ValueError):
        return f"See the evaluation results in the [GitHub repo]({REPO_URL})."
