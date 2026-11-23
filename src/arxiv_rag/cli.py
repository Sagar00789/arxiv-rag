"""Command line: ingest -> index -> make-testset -> eval -> ask."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import typer

from .config import Settings

app = typer.Typer(add_completion=False, help="Q&A over arXiv papers, with retrieval evals.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# Hide per-request HTTP logs from HuggingFace / LiteLLM; keep our own progress logs.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "LiteLLM", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def friendly_errors(fn):
    """Show LLM config problems as one clear line instead of a long traceback."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from .generate import LLMError

        try:
            return fn(*args, **kwargs)
        except LLMError as e:
            typer.secho(f"LLM error: {e}", fg="red", err=True)
            raise typer.Exit(1) from None

    return wrapper


DEFAULT_QUERY = 'abs:"retrieval-augmented generation" OR abs:"LLM agents"'


@app.command()
def ingest(
    query: str = typer.Option(DEFAULT_QUERY, help="arXiv search query"),
    max_results: int = typer.Option(50, help="Number of papers to download"),
):
    """Download papers from arXiv and chunk them into data/chunks.jsonl."""
    from .ingest import build_corpus

    s = Settings()
    n_papers, n_chunks = build_corpus(query, max_results, s.data_dir, s.chunk_size, s.chunk_overlap)
    typer.echo(f"{n_papers} papers -> {n_chunks} chunks in {s.chunks_path}")


@app.command()
def index():
    """Embed all chunks and (re)build the vector index."""
    from .chunking import load_chunks
    from .embed import SentenceTransformerEmbedder
    from .retrieve import index_chunks
    from .store import VectorStore

    s = Settings()
    chunks = load_chunks(s.chunks_path) if s.chunks_path.exists() else []
    if not chunks:
        typer.echo(f"No chunks in {s.chunks_path}. Run `arxiv-rag ingest` first.", err=True)
        raise typer.Exit(1)
    embedder = SentenceTransformerEmbedder(s.embed_model)
    store = VectorStore(s.qdrant_path, s.collection, embedder.dim)
    typer.echo(f"Indexed {index_chunks(chunks, embedder, store)} chunks with {s.embed_model}")


@app.command("check-llm")
@friendly_errors
def check_llm_cmd():
    """Make one tiny LLM call to confirm the model name and API key work."""
    from .generate import check_llm

    s = Settings()
    reply = check_llm(s.llm_model).strip()
    typer.secho(f"OK: {s.llm_model} replied {reply[:40]!r}", fg="green")


@app.command("make-testset")
@friendly_errors
def make_testset_cmd(
    n: int = typer.Option(50, help="Number of questions"),
    out: Path = typer.Option(Path("eval/testset.jsonl")),
    rpm: float = typer.Option(12, help="Max LLM requests per minute (free tiers allow ~15)"),
    fresh: bool = typer.Option(False, help="Delete the existing test set and start over"),
):
    """Generate a synthetic test set with the LLM (resumable). Review it by hand after."""
    from .chunking import load_chunks
    from .evaluation import load_testset, make_testset
    from .generate import LLMError, set_rate_limit

    s = Settings()
    chunks = load_chunks(s.chunks_path) if s.chunks_path.exists() else []
    if not chunks:
        typer.echo(f"No chunks in {s.chunks_path}. Run `arxiv-rag ingest` first.", err=True)
        raise typer.Exit(1)
    if fresh and out.exists():
        out.unlink()
    set_rate_limit(rpm)
    try:
        items = make_testset(chunks, n, s.llm_model, out)
    except LLMError:
        saved = len(load_testset(out)) if out.exists() else 0
        typer.echo(f"{saved} questions saved in {out}. Re-run the same command to continue.")
        raise
    if not items:
        typer.echo("The LLM produced no usable questions. Check LLM_MODEL and API key.", err=True)
        raise typer.Exit(1)
    typer.echo(f"{len(items)} questions in {out}")


@app.command("eval")
def eval_cmd(
    retriever: list[str] = typer.Option(
        None, "--retriever", "-r", help="dense | bm25 | hybrid | hybrid-rerank (repeatable)"
    ),
    name: str = typer.Option(None, help="Experiment name (default: the retriever name)"),
    testset: Path = typer.Option(Path("eval/testset.jsonl")),
    k: int = typer.Option(5),
):
    """Score retrieval on the test set and append a row per retriever to eval/RESULTS.md."""
    from .evaluation import evaluate_retrieval, load_testset, record_results
    from .pipeline import build_retriever

    items = load_testset(testset) if testset.exists() else []
    if not items:
        typer.echo(f"No questions in {testset}. Run `arxiv-rag make-testset` first.", err=True)
        raise typer.Exit(1)
    s = Settings()
    for r in retriever or [s.retriever]:
        results = evaluate_retrieval(build_retriever(s, r), items, k=k)
        exp = name if name and len(retriever or []) <= 1 else r
        record_results(results, exp, testset.parent)
        summary = "  ".join(f"{key}={v}" for key, v in results.items() if key not in ("n", "k"))
        typer.secho(f"{exp:>14}: {summary}", fg="green")
    typer.echo(f"Logged to {testset.parent / 'RESULTS.md'}")


@app.command()
@friendly_errors
def ask(
    question: str,
    k: int = typer.Option(5),
    retriever: str = typer.Option(None, "--retriever", "-r", help="Override RETRIEVER"),
):
    """Ask a question and print the cited answer."""
    from .pipeline import RAGPipeline

    ans = RAGPipeline.from_settings(retriever=retriever).ask(question, k)
    typer.echo(ans.answer + "\n\nSources:")
    for i, h in enumerate(ans.sources, start=1):
        typer.echo(f"  [{i}] {h.chunk.title} (arXiv:{h.chunk.paper_id}) score={h.score:.3f}")


if __name__ == "__main__":
    app()
