"""Web demo (Gradio). Run locally with `arxiv-rag ui`; deployed to Hugging Face Spaces."""

from __future__ import annotations

import html
import time
from collections.abc import Callable
from pathlib import Path

from .config import Settings
from .pipeline import Answer, RAGPipeline

REPO_URL = "https://github.com/Sagar00789/arxiv-rag"

RETRIEVER_CHOICES = [
    ("Hybrid, 80% BM25 (default, best all-rounder)", "hybrid@0.8"),
    ("BM25 keywords only (fastest)", "bm25"),
    ("Dense embeddings only", "dense"),
    ("Hybrid, equal weights", "hybrid"),
    ("Hybrid + cross-encoder rerank (slow on CPU)", "hybrid@0.8-rerank"),
]

EXAMPLES = [
    "How does Self-RAG decide when to retrieve?",
    "What are common ways to evaluate retrieval-augmented generation systems?",
    "How do multi-agent systems handle conflicting evidence from retrieved documents?",
    "What is reciprocal rank fusion?",
]

PipelineFactory = Callable[[str], RAGPipeline]


def default_factory() -> PipelineFactory:
    cache: dict[str, RAGPipeline] = {}
    settings = Settings()

    def get(retriever: str) -> RAGPipeline:
        if retriever not in cache:
            cache[retriever] = RAGPipeline.from_settings(settings, retriever=retriever)
        return cache[retriever]

    return get


def format_sources(ans: Answer) -> str:
    if not ans.sources:
        return "_No passages retrieved._"
    parts = []
    for i, h in enumerate(ans.sources, start=1):
        c = h.chunk
        snippet = html.escape(c.text[:600] + ("…" if len(c.text) > 600 else ""))
        parts.append(
            f"**[{i}] [{html.escape(c.title)}](https://arxiv.org/abs/{c.paper_id})**  \n"
            f"<sub>arXiv:{c.paper_id} · passage {c.position} · score {h.score:.3f}</sub>\n"
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
    footer = f"\n\n<sub>{retriever} · {len(ans.sources)} passages · {took:.1f}s</sub>"
    return ans.answer + footer, format_sources(ans)


def results_markdown(readme: Path = Path("README.md")) -> str:
    """The results section of the README, so the demo shows the evaluation too."""
    try:
        text = readme.read_text(encoding="utf-8")
        start = text.index("## Results")
        end = text.index("## Quickstart", start)
        return text[start:end]
    except (OSError, ValueError):
        return f"See the evaluation results in the [GitHub repo]({REPO_URL})."


def build_app(factory: PipelineFactory | None = None):
    import gradio as gr

    factory = factory or default_factory()

    with gr.Blocks(title="arxiv-rag: cited answers from AI research papers") as app:
        gr.Markdown(
            "# arxiv-rag\n"
            "Ask about **retrieval-augmented generation and LLM agents**. Answers come only from "
            "49 arXiv papers and cite the passages they use. "
            f"[Code, tests and evaluation on GitHub]({REPO_URL})."
        )
        with gr.Tab("Ask"):
            question = gr.Textbox(
                label="Question", placeholder="e.g. How does Self-RAG decide when to retrieve?"
            )
            retriever = gr.Dropdown(RETRIEVER_CHOICES, value="hybrid@0.8", label="Retriever")
            ask = gr.Button("Ask", variant="primary")
            answer = gr.Markdown(label="Answer")
            gr.Markdown("### Sources")
            sources = gr.Markdown()
            gr.Examples(EXAMPLES, inputs=question)

            def run(q: str, r: str) -> tuple[str, str]:
                return answer_question(q, r, factory)

            ask.click(run, [question, retriever], [answer, sources])
            question.submit(run, [question, retriever], [answer, sources])
        with gr.Tab("How well does it work?"):
            gr.Markdown(results_markdown())
    return app


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    build_app().launch(server_name=host, server_port=port)
