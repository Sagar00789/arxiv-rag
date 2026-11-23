"""Command line: ingest -> index -> make-testset -> eval -> ask."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from .config import Settings

app = typer.Typer(add_completion=False, help="Q&A over arXiv papers, with retrieval evals.")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
# Hide per-request HTTP logs from HuggingFace / LiteLLM; keep our own progress logs.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "LiteLLM", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

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


@app.command("make-testset")
def make_testset_cmd(
    n: int = typer.Option(50, help="Number of questions"),
    out: Path = typer.Option(Path("eval/testset.jsonl")),
):
    """Generate a synthetic test set with the LLM. Review it by hand before trusting it."""
    from .chunking import load_chunks
    from .evaluation import make_testset, save_testset

    s = Settings()
    chunks = load_chunks(s.chunks_path) if s.chunks_path.exists() else []
    if not chunks:
        typer.echo(f"No chunks in {s.chunks_path}. Run `arxiv-rag ingest` first.", err=True)
        raise typer.Exit(1)
    items = make_testset(chunks, n, s.llm_model)
    if not items:
        typer.echo("The LLM produced no usable questions. Check LLM_MODEL and API key.", err=True)
        raise typer.Exit(1)
    save_testset(items, out)
    typer.echo(f"Wrote {len(items)} questions to {out}")


@app.command("eval")
def eval_cmd(
    name: str = typer.Option("baseline-dense", help="Experiment name for the results table"),
    testset: Path = typer.Option(Path("eval/testset.jsonl")),
    k: int = typer.Option(5),
):
    """Score retrieval on the test set and append a row to eval/RESULTS.md."""
    from .evaluation import evaluate_retrieval, load_testset, record_results
    from .pipeline import RAGPipeline

    pipe = RAGPipeline.from_settings()
    results = evaluate_retrieval(pipe.retriever, load_testset(testset), k=k)
    path = record_results(results, name, testset.parent)
    for key, v in results.items():
        typer.echo(f"{key:>16}: {v}")
    typer.echo(f"Saved {path}")


@app.command()
def ask(question: str, k: int = typer.Option(5)):
    """Ask a question and print the cited answer."""
    from .pipeline import RAGPipeline

    ans = RAGPipeline.from_settings().ask(question, k)
    typer.echo(ans.answer + "\n\nSources:")
    for i, h in enumerate(ans.sources, start=1):
        typer.echo(f"  [{i}] {h.chunk.title} (arXiv:{h.chunk.paper_id}) score={h.score:.3f}")


if __name__ == "__main__":
    app()
