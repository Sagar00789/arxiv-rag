"""Wires retrieval and generation together."""

from __future__ import annotations

import functools
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .generate import generate_answer
from .retrieve import Hit, Retriever, Scorer

RETRIEVERS = ("dense", "bm25", "hybrid", "hybrid-rerank", "bm25-rerank")
# Names: <dense|bm25|hybrid>[@<bm25 weight>][-rerank], e.g. "hybrid@0.8", "hybrid@0.8-rerank"
_NAME = re.compile(r"^(dense|bm25|hybrid)(?:@(0(?:\.\d+)?|1(?:\.0+)?|\.\d+))?(-rerank)?$")


def parse_retriever(name: str) -> tuple[str, float, bool]:
    """Return (base, bm25_weight, rerank). Weight only applies to hybrid (default 0.5)."""
    m = _NAME.match(name)
    if not m or (m.group(2) and m.group(1) != "hybrid"):
        raise ValueError(
            f"Unknown retriever {name!r}. Use dense, bm25 or hybrid, optionally "
            "hybrid@<bm25 weight 0-1> and/or a -rerank suffix, e.g. hybrid@0.8-rerank"
        )
    return m.group(1), float(m.group(2) or 0.5), bool(m.group(3))


class IndexMissingError(RuntimeError):
    pass


@functools.cache
def _dense(
    embed_model: str, qdrant_path: str, collection: str, backend: str = "sentence-transformers"
) -> Retriever:
    # Cached: embedded Qdrant locks its folder, so open it once per process.
    from .embed import make_embedder
    from .retrieve import DenseRetriever
    from .store import VectorStore

    embedder = make_embedder(embed_model, backend)
    store = VectorStore(qdrant_path, collection, embedder.dim)
    if not store.exists() or store.count() == 0:
        raise IndexMissingError(
            f"No index for embedding model {embed_model} (collection '{collection}'). "
            f"Run: arxiv-rag index --embed-model {embed_model}"
        )
    return DenseRetriever(embedder, store)


@functools.cache
def _cross_encoder(model_name: str):
    from .retrieve import cross_encoder_scorer

    return cross_encoder_scorer(model_name)


@functools.cache
def _bm25(chunks_path: str) -> Retriever:
    from .chunking import load_chunks
    from .retrieve import BM25Retriever

    return BM25Retriever(load_chunks(Path(chunks_path)))


def assemble_retriever(
    name: str,
    dense: Callable[[], Retriever],
    bm25: Callable[[], Retriever],
    scorer: Callable[[], Scorer],
) -> Retriever:
    """Build a named retriever from its parts. Parts are factories, so only what the name
    needs gets loaded."""
    from .retrieve import HybridRetriever, RerankRetriever

    base, w, rerank = parse_retriever(name)
    if base == "dense":
        r = dense()
    elif base == "bm25":
        r = bm25()
    else:
        r = HybridRetriever([dense(), bm25()], weights=[1 - w, w])
    return RerankRetriever(r, scorer()) if rerank else r


def build_retriever(s: Settings, name: str) -> Retriever:
    """Build a retriever by name from the on-disk index."""
    return assemble_retriever(
        name,
        dense=lambda: _dense(s.embed_model, s.qdrant_path, s.dense_collection, s.embed_backend),
        bm25=lambda: _bm25(str(s.chunks_path)),
        scorer=lambda: _cross_encoder(s.reranker_model),
    )


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Hit] = field(default_factory=list)
    retrieval_s: float = 0.0
    generation_s: float = 0.0


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm_model: str, top_k: int = 5):
        self.retriever = retriever
        self.llm_model = llm_model
        self.top_k = top_k

    @classmethod
    def from_settings(
        cls, settings: Settings | None = None, retriever: str | None = None
    ) -> RAGPipeline:
        s = settings or Settings()
        return cls(build_retriever(s, retriever or s.retriever), s.llm_model, s.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, k or self.top_k)
        t1 = time.perf_counter()
        text = generate_answer(question, hits, self.llm_model)
        return Answer(question, text, hits, t1 - t0, time.perf_counter() - t1)
