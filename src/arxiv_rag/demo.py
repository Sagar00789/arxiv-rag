"""Builds everything the hosted demo needs, in memory, from the committed list of arXiv IDs.

Free hosting has little RAM and no persistent disk, so the demo:
- fetches the same 49 papers as the evaluation (eval/paper_ids.txt) on first start,
- embeds them with FastEmbed (ONNX, no PyTorch) into an in-memory Qdrant collection,
- serves the same retrievers as the CLI, minus the cross-encoder reranker.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk, load_chunks
from .config import Settings
from .embed import Embedder
from .pipeline import RAGPipeline, assemble_retriever
from .retrieve import BM25Retriever, DenseRetriever, Hit, Retriever
from .store import VectorStore

Progress = Callable[[str], None]


class LockedRetriever:
    """Serialises access to a shared retriever: the demo serves several browser sessions
    from threads, and the in-memory index isn't guaranteed thread-safe."""

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


def ensure_corpus(s: Settings, ids_file: Path, progress: Progress) -> list[Chunk]:
    """Download and chunk the papers unless this container already has them."""
    if not s.chunks_path.exists():
        from .ingest import build_corpus, read_id_list

        ids = read_id_list(ids_file)
        progress(f"Downloading {len(ids)} papers from arXiv (first start only, ~4 min)…")
        build_corpus("", len(ids), s.data_dir, s.chunk_size, s.chunk_overlap, id_list=ids)
    return load_chunks(s.chunks_path)


def _vectors_path(s: Settings) -> Path:
    return s.data_dir / f"vectors_{s.dense_collection}.json"


def load_cached_vectors(s: Settings, chunks: list[Chunk], dim: int) -> list[list[float]] | None:
    """Embeddings saved by an earlier start, so code updates and restarts skip re-embedding.
    Ignored unless they match the current chunks and model exactly."""
    path = _vectors_path(s)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    ids = [c.chunk_id for c in chunks]
    vecs = data.get("vectors", [])
    if data.get("chunk_ids") != ids or not vecs or len(vecs[0]) != dim:
        return None
    return vecs


def save_cached_vectors(s: Settings, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    path = _vectors_path(s)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"chunk_ids": [c.chunk_id for c in chunks], "vectors": vectors}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)  # atomic: a crash mid-write never leaves a broken cache


def build_demo(
    s: Settings,
    ids_file: Path,
    progress: Progress = lambda msg: None,
    embedder_factory: Callable[[str], Embedder] | None = None,
) -> Demo:
    chunks = ensure_corpus(s, ids_file, progress)
    if embedder_factory is None:
        from .embed import FastEmbedEmbedder

        embedder_factory = FastEmbedEmbedder
    embedder = embedder_factory(s.embed_model)
    store = VectorStore(":memory:", s.dense_collection, embedder.dim)
    vectors = load_cached_vectors(s, chunks, embedder.dim)
    if vectors is None:
        progress(f"Embedding {len(chunks):,} passages with {s.embed_model} (first start only)…")
        vectors = embedder.embed_documents([c.text for c in chunks])
        save_cached_vectors(s, chunks, vectors)
    else:
        progress(f"Loaded {len(chunks):,} cached embeddings.")
    store.reset()
    store.upsert(chunks, vectors)
    progress("Building the keyword index…")
    return Demo(
        chunks=chunks,
        dense=DenseRetriever(embedder, store),
        bm25=BM25Retriever(chunks),
        llm_model=s.llm_model,
        top_k=s.top_k,
    )
