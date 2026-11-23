"""Command line: ingest -> index -> make-testset -> eval -> ask."""

from __future__ import annotations

import dataclasses
import functools
import logging
import time
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
        from .pipeline import IndexMissingError

        try:
            return fn(*args, **kwargs)
        except LLMError as e:
            typer.secho(f"LLM error: {e}", fg="red", err=True)
            raise typer.Exit(1) from None
        except IndexMissingError as e:
            typer.secho(str(e), fg="red", err=True)
            raise typer.Exit(1) from None

    return wrapper


DEFAULT_QUERY = 'abs:"retrieval-augmented generation" OR abs:"LLM agents"'


@app.command()
def ingest(
    query: str = typer.Option(DEFAULT_QUERY, help="arXiv search query"),
    max_results: int = typer.Option(50, help="Number of papers to download"),
    ids_file: Path = typer.Option(
        None, help="File of exact arXiv IDs to fetch instead of searching (e.g. eval/paper_ids.txt)"
    ),
):
    """Download papers from arXiv and chunk them into data/chunks.jsonl."""
    from .ingest import build_corpus, read_id_list

    s = Settings()
    ids = read_id_list(ids_file) if ids_file else None
    n_papers, n_chunks = build_corpus(
        query, max_results, s.data_dir, s.chunk_size, s.chunk_overlap, id_list=ids
    )
    typer.echo(f"{n_papers} papers -> {n_chunks} chunks in {s.chunks_path}")


EMBED_HELP = "Embedding model(s), e.g. BAAI/bge-base-en-v1.5 (repeatable; default EMBED_MODEL)"


def _check_retrievers(names: list[str]) -> None:
    """Fail fast on a typo, before loading any model."""
    from .pipeline import parse_retriever

    for r in names:
        try:
            parse_retriever(r)
        except ValueError as e:
            typer.secho(str(e), fg="red", err=True)
            raise typer.Exit(1) from None


def _with_models(s: Settings, models: list[str] | None) -> list[Settings]:
    return [dataclasses.replace(s, embed_model=m) for m in models] if models else [s]


@app.command()
def index(embed_model: list[str] = typer.Option(None, "--embed-model", "-e", help=EMBED_HELP)):
    """Embed all chunks and (re)build the vector index, one collection per embedding model."""
    from .chunking import load_chunks
    from .embed import make_embedder
    from .retrieve import index_chunks
    from .store import VectorStore

    base = Settings()
    chunks = load_chunks(base.chunks_path) if base.chunks_path.exists() else []
    if not chunks:
        typer.echo(f"No chunks in {base.chunks_path}. Run `arxiv-rag ingest` first.", err=True)
        raise typer.Exit(1)
    for s in _with_models(base, embed_model):
        t0 = time.perf_counter()
        embedder = make_embedder(s.embed_model, s.embed_backend)
        store = VectorStore(s.qdrant_path, s.dense_collection, embedder.dim)
        n = index_chunks(chunks, embedder, store)
        typer.secho(
            f"Indexed {n} chunks with {s.embed_model} ({embedder.dim} dims) "
            f"into '{s.dense_collection}' in {time.perf_counter() - t0:.0f}s",
            fg="green",
        )


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


@app.command("paraphrase-testset")
@friendly_errors
def paraphrase_testset_cmd(
    src: Path = typer.Option(Path("eval/testset.jsonl"), "--in"),
    out: Path = typer.Option(Path("eval/testset_paraphrased.jsonl")),
    rpm: float = typer.Option(12, help="Max LLM requests per minute"),
):
    """Rewrite each test question so it doesn't reuse the passage's wording (resumable)."""
    from .chunking import load_chunks
    from .evaluation import load_testset, mean_overlap, paraphrase_testset
    from .generate import LLMError, set_rate_limit

    s = Settings()
    chunks = {c.chunk_id: c for c in load_chunks(s.chunks_path)}
    items = load_testset(src)
    set_rate_limit(rpm)
    try:
        done = paraphrase_testset(items, chunks, s.llm_model, out)
    except LLMError:
        typer.echo(f"Progress saved in {out}. Re-run the same command to continue.")
        raise
    typer.echo(f"{len(done)}/{len(items)} questions paraphrased into {out}")
    typer.secho(
        f"Word overlap with gold passage: {mean_overlap(items, chunks)} (original) -> "
        f"{mean_overlap(done, chunks)} (paraphrased)",
        fg="green",
    )


@app.command("eval")
@friendly_errors
def eval_cmd(
    retriever: list[str] = typer.Option(
        None,
        "--retriever",
        "-r",
        help="dense | bm25 | hybrid[@w] | any of these + '-rerank' (repeatable)",
    ),
    embed_model: list[str] = typer.Option(None, "--embed-model", "-e", help=EMBED_HELP),
    name: str = typer.Option(None, help="Experiment name (default: the retriever name)"),
    testset: Path = typer.Option(Path("eval/testset.jsonl")),
    k: int = typer.Option(5),
):
    """Score retrieval on the test set: one row per (embedding model, retriever) in RESULTS.md."""
    from .chunking import load_chunks
    from .evaluation import evaluate_retrieval, load_testset, mean_overlap, record_results
    from .pipeline import build_retriever, parse_retriever

    items = load_testset(testset) if testset.exists() else []
    if not items:
        typer.echo(f"No questions in {testset}. Run `arxiv-rag make-testset` first.", err=True)
        raise typer.Exit(1)
    s = Settings()
    names = retriever or [s.retriever]
    _check_retrievers(names)

    label = testset.stem.removeprefix("testset").strip("_-")  # "" or e.g. "paraphrased"
    chunks = {c.chunk_id: c for c in load_chunks(s.chunks_path)} if s.chunks_path.exists() else {}
    overlap = mean_overlap(items, chunks)
    typer.echo(f"Test set {testset}: {len(items)} questions, word overlap {overlap}")
    from .config import DEFAULT_EMBED_MODEL, model_short_name

    runs = [(sm, r) for sm in _with_models(s, embed_model) for r in names]
    for sm, r in runs:
        uses_dense = parse_retriever(r)[0] != "bm25"
        if not uses_dense and sm.embed_model != (embed_model or [s.embed_model])[0]:
            continue  # bm25 doesn't depend on the embedding model; score it once
        results = evaluate_retrieval(build_retriever(sm, r), items, k=k)
        results["word_overlap"] = overlap
        results["embed_model"] = sm.embed_model if uses_dense else None
        exp = name if name and len(runs) == 1 else r
        if uses_dense and sm.embed_model != DEFAULT_EMBED_MODEL:
            exp = f"{exp} · {model_short_name(sm.embed_model)}"
        if label:
            exp = f"{exp} [{label}]"
        record_results(results, exp, testset.parent)
        summary = "  ".join(
            f"{key}={v}"
            for key, v in results.items()
            if key not in ("n", "k", "word_overlap", "embed_model")
        )
        typer.secho(f"{exp:>44}: {summary}", fg="green")
    typer.echo(f"Logged to {testset.parent / 'RESULTS.md'}")


@app.command("eval-answers")
@friendly_errors
def eval_answers_cmd(
    retriever: str = typer.Option(None, "--retriever", "-r", help="Default: RETRIEVER"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Default: EMBED_MODEL"),
    testset: Path = typer.Option(Path("eval/testset.jsonl")),
    n: int = typer.Option(25, help="Questions to grade (2 LLM calls each)"),
    judge_model: str = typer.Option(
        None, help="LLM used as judge (default: LLM_MODEL; a different model avoids self-grading)"
    ),
    rpm: float = typer.Option(12, help="Max LLM requests per minute"),
    k: int = typer.Option(5),
):
    """Answer test questions and grade them with an LLM judge (resumable)."""
    import re as _re

    from .answer_eval import evaluate_answers, record_answer_results, summarize
    from .config import DEFAULT_EMBED_MODEL, model_short_name
    from .evaluation import load_testset
    from .generate import LLMError, set_rate_limit
    from .pipeline import build_retriever, parse_retriever

    s = _with_models(Settings(), [embed_model] if embed_model else None)[0]
    r = retriever or s.retriever
    _check_retrievers([r])
    items = load_testset(testset)[:n]
    judge = judge_model or s.llm_model

    exp = r
    if parse_retriever(r)[0] != "bm25" and s.embed_model != DEFAULT_EMBED_MODEL:
        exp += f" · {model_short_name(s.embed_model)}"
    label = testset.stem.removeprefix("testset").strip("_-")
    if label:
        exp += f" [{label}]"
    slug = _re.sub(r"[^\w.@-]+", "_", exp).strip("_")
    log_path = testset.parent / "answers" / f"{slug}.jsonl"

    if judge == s.llm_model:
        typer.secho(
            f"Note: {judge} is grading its own answers. Pass --judge-model to use a different one.",
            fg="yellow",
        )
    set_rate_limit(rpm)
    try:
        judged = evaluate_answers(build_retriever(s, r), items, s.llm_model, judge, log_path, k)
    except LLMError:
        typer.echo(f"Progress saved in {log_path}. Re-run the same command to continue.")
        raise
    summary = summarize(judged)
    record_answer_results(summary, exp, testset.parent)
    for key, v in summary.items():
        typer.secho(f"{key:>32}: {v}", fg="green")
    typer.echo(f"Details: {log_path}   Summary: {testset.parent / 'ANSWERS.md'}")


@app.command()
@friendly_errors
def ask(
    question: str,
    k: int = typer.Option(5),
    retriever: str = typer.Option(None, "--retriever", "-r", help="Override RETRIEVER"),
    embed_model: str = typer.Option(None, "--embed-model", "-e", help="Override EMBED_MODEL"),
):
    """Ask a question and print the cited answer."""
    from .pipeline import RAGPipeline

    s = _with_models(Settings(), [embed_model] if embed_model else None)[0]
    ans = RAGPipeline.from_settings(s, retriever=retriever).ask(question, k)
    typer.echo(ans.answer + "\n\nSources:")
    for i, h in enumerate(ans.sources, start=1):
        typer.echo(f"  [{i}] {h.chunk.title} (arXiv:{h.chunk.paper_id}) score={h.score:.3f}")


@app.command()
def ui(port: int = typer.Option(8501)):
    """Launch the web demo locally (needs the `demo` extra: pip install -e ".[all,demo]")."""
    import subprocess
    import sys

    app_path = Path(__file__).resolve().parents[2] / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
