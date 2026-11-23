"""Builds everything the hosted demo needs from the committed list of arXiv IDs, within ~1 GB RAM.

Free hosting gives about 1 GB of RAM and no GPU, so the demo:
- does the heavy one-time work (download PDFs, extract text, embed 1,681 passages) in a
  **separate process**, so PyMuPDF and the model's working memory are returned to the OS
  when it exits; the embeddings are saved to disk and reused on every later start;
- embeds in small batches: attention memory grows with batch size (a batch of 64
  passages at 512 tokens needs ~800 MB for one layer's attention scores; a batch of 4, ~50 MB);
- searches the 1,681 vectors with plain NumPy (exact, ~1 ms) instead of loading the
  Qdrant client (~110 MB); the evaluation pipeline still uses Qdrant;
- serves the same retrievers as the CLI, minus the cross-encoder reranker.
Run the build step by hand with: python -m arxiv_rag.demo
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, load_chunks
from .config import Settings
from .embed import Embedder
from .pipeline import RAGPipeline, assemble_retriever
from .retrieve import BM25Retriever, DenseRetriever, Hit, Retriever

Progress = Callable[[str], None]
EMBED_BATCH = 4  # keeps peak attention memory around 50 MB (see module docstring)


# ---------- in-memory vector search ----------


class ArrayStore:
    """Exact cosine search over normalized vectors with NumPy. For a few thousand vectors this
    is as fast as a vector database and uses a fraction of the memory."""

    def __init__(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]):
        import numpy as np

        if len(chunks) != len(vectors):
            raise ValueError("need one vector per chunk")
        self.chunks = list(chunks)
        self.matrix = np.asarray(vectors, dtype=np.float32)

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        import numpy as np

        scores = self.matrix @ np.asarray(vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]


class LockedRetriever:
    """Serialises access to a shared retriever: the demo serves several browser sessions
    from threads."""

    def __init__(self, inner: Retriever, lock: threading.Lock):
        self.inner = inner
        self.lock = lock

    def retrieve(self, query: str, k: int) -> list[Hit]:
        with self.lock:
            return self.inner.retrieve(query, k)


@dataclass
class Demo:
    chunks: list[Chunk]
    dense: Retriever
    bm25: Retriever
    llm_model: str
    top_k: int = 5
    _pipelines: dict[str, RAGPipeline] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def pipeline(self, retriever: str) -> RAGPipeline:
        if retriever not in self._pipelines:
            r = assemble_retriever(
                retriever,
                dense=lambda: self.dense,
                bm25=lambda: self.bm25,
                scorer=lambda: (_ for _ in ()).throw(
                    ValueError("Reranking isn't available in the hosted demo.")
                ),
            )
            self._pipelines[retriever] = RAGPipeline(
                LockedRetriever(r, self._lock), self.llm_model, self.top_k
            )
        return self._pipelines[retriever]


# ---------- embedding cache ----------


def _vectors_path(s: Settings) -> Path:
    return s.data_dir / f"vectors_{s.dense_collection}.json"


def load_cached_vectors(
    s: Settings, chunks: list[Chunk], dim: int | None = None
) -> list[list[float]] | None:
    """Embeddings saved by an earlier build. Ignored unless they match the current chunks
    (and dimension, when given) exactly."""
    try:
        data = json.loads(_vectors_path(s).read_text())
    except (OSError, ValueError):
        return None
    vecs = data.get("vectors", [])
    if data.get("chunk_ids") != [c.chunk_id for c in chunks] or not vecs:
        return None
    if dim is not None and len(vecs[0]) != dim:
        return None
    return vecs


def save_cached_vectors(s: Settings, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    path = _vectors_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunk_ids": [c.chunk_id for c in chunks], "vectors": vectors}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a broken cache


# ---------- the heavy, one-time build ----------


def ensure_corpus(s: Settings, ids_file: Path, progress: Progress) -> list[Chunk]:
    """Download and chunk the papers unless they're already on disk."""
    if not s.chunks_path.exists():
        from .ingest import build_corpus, read_id_list

        ids = read_id_list(ids_file)
        progress(f"Downloading {len(ids)} papers from arXiv (first start only, ~4 min)…")
        build_corpus("", len(ids), s.data_dir, s.chunk_size, s.chunk_overlap, id_list=ids)
    return load_chunks(s.chunks_path)


def build_vectors(
    s: Settings, ids_file: Path, progress: Progress, embedder: Embedder
) -> tuple[list[Chunk], list[list[float]]]:
    """Corpus + embeddings, from the disk cache when possible."""
    chunks = ensure_corpus(s, ids_file, progress)
    vectors = load_cached_vectors(s, chunks, embedder.dim)
    if vectors is None:
        progress(f"Embedding {len(chunks):,} passages with {s.embed_model} (first start only)…")
        vectors = []
        for i in range(0, len(chunks), 256):
            vectors += embedder.embed_documents([c.text for c in chunks[i : i + 256]])
            progress(f"  embedded {min(i + 256, len(chunks)):,}/{len(chunks):,}")
        save_cached_vectors(s, chunks, vectors)
    else:
        progress(f"Loaded {len(chunks):,} cached embeddings.")
    return chunks, vectors


def run_build_subprocess(ids_file: Path, progress: Progress, cmd: list[str] | None = None) -> None:
    """Run the build in a child process and stream its output. The child inherits the
    environment, so it uses the same settings."""
    cmd = cmd or [sys.executable, "-m", "arxiv_rag.demo", str(ids_file)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-15:]
            progress(line)
    if proc.wait() != 0:
        raise RuntimeError("Building the demo index failed:\n" + "\n".join(tail))


def _cache_ready(s: Settings) -> bool:
    return s.chunks_path.exists() and load_cached_vectors(s, load_chunks(s.chunks_path)) is not None


# ---------- what the web app calls ----------


def build_demo(
    s: Settings,
    ids_file: Path,
    progress: Progress = lambda msg: None,
    embedder_factory: Callable[[str], Embedder] | None = None,
) -> Demo:
    """Ready-to-serve demo. With the default (FastEmbed) embedder, the heavy build runs in a
    child process and this process only loads the result."""
    if embedder_factory is None:
        from .embed import FastEmbedEmbedder

        if not _cache_ready(s):
            run_build_subprocess(ids_file, progress)
        embedder_factory = FastEmbedEmbedder
    embedder = embedder_factory(s.embed_model)  # still needed to embed each question
    chunks, vectors = build_vectors(s, ids_file, progress, embedder)
    progress("Building the keyword index…")
    return Demo(
        chunks=chunks,
        dense=DenseRetriever(embedder, ArrayStore(chunks, vectors)),
        bm25=BM25Retriever(chunks),
        llm_model=s.llm_model,
        top_k=s.top_k,
    )


def _main() -> None:
    import logging

    from .embed import FastEmbedEmbedder

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = Settings()
    ids_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/paper_ids.txt")
    embedder = FastEmbedEmbedder(s.embed_model, batch_size=EMBED_BATCH)
    build_vectors(s, ids_file, lambda m: print(m, flush=True), embedder)
    print("Build finished.", flush=True)


if __name__ == "__main__":
    _main()
