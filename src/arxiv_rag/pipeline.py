"""Wires retrieval and generation together."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings
from .generate import generate_answer
from .retrieve import Hit, Retriever

RETRIEVERS = ("dense", "bm25", "hybrid", "hybrid-rerank")


@functools.cache
def _dense(embed_model: str, qdrant_path: str, collection: str) -> Retriever:
    # Cached: embedded Qdrant locks its folder, so open it once per process.
    from .embed import SentenceTransformerEmbedder
    from .retrieve import DenseRetriever
    from .store import VectorStore

    embedder = SentenceTransformerEmbedder(embed_model)
    return DenseRetriever(embedder, VectorStore(qdrant_path, collection, embedder.dim))


@functools.cache
def _bm25(chunks_path: str) -> Retriever:
    from .chunking import load_chunks
    from .retrieve import BM25Retriever

    return BM25Retriever(load_chunks(Path(chunks_path)))


def build_retriever(s: Settings, name: str) -> Retriever:
    """Build a retriever by name. Heavy pieces are only loaded when that name needs them."""
    from .retrieve import HybridRetriever, RerankRetriever

    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever {name!r}; choose from {', '.join(RETRIEVERS)}")

    def dense() -> Retriever:
        return _dense(s.embed_model, s.qdrant_path, s.collection)

    def bm25() -> Retriever:
        return _bm25(str(s.chunks_path))

    if name == "dense":
        return dense()
    if name == "bm25":
        return bm25()
    hybrid = HybridRetriever([dense(), bm25()])
    if name == "hybrid":
        return hybrid
    from .retrieve import cross_encoder_scorer

    return RerankRetriever(hybrid, cross_encoder_scorer(s.reranker_model))


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Hit] = field(default_factory=list)


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
        hits = self.retriever.retrieve(question, k or self.top_k)
        return Answer(question, generate_answer(question, hits, self.llm_model), hits)
