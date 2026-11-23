"""Retrievers. The baseline is dense-only; hybrid (BM25 + dense) and reranking come next."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chunking import Chunk
from .embed import Embedder
from .store import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int) -> list[Hit]:
        vec = self.embedder.embed_query(query)
        return [Hit(chunk=c, score=s) for c, s in self.store.search(vec, k)]


def index_chunks(
    chunks: list[Chunk], embedder: Embedder, store: VectorStore, batch: int = 256
) -> int:
    store.reset()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        store.upsert(part, embedder.embed_documents([c.text for c in part]))
    return store.count()
