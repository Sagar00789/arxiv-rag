"""Retrievers: dense (embeddings), BM25 (keywords), hybrid fusion, and cross-encoder reranking.

Each one has the same `retrieve(query, k)` interface, so the eval harness can compare them
like for like.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
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


# ---------- BM25 (keyword) ----------

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")  # keeps "main-rag", "gpt-4", "3.5"
_STOPWORDS = frozenset(
    "a an and are as at be by does for from has have how in is it its of on or that the "
    "their this to was what when which who why with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Classic keyword ranking. Strong exactly where embeddings are weak: rare names,
    acronyms and numbers."""

    def __init__(self, chunks: Sequence[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text + " " + c.title) for c in self.chunks])

    def retrieve(self, query: str, k: int) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top if scores[i] > 0]


# ---------- hybrid (reciprocal rank fusion) ----------


def reciprocal_rank_fusion(result_lists: Sequence[list[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Merge rankings by summing 1 / (rrf_k + rank). Uses ranks only, so the very
    different score scales of BM25 and cosine similarity don't need calibrating."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            chunks[cid] = h.chunk
    best = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [Hit(chunks[cid], scores[cid]) for cid in best]


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], candidates: int = 30):
        self.retrievers = list(retrievers)
        self.candidates = candidates

    def retrieve(self, query: str, k: int) -> list[Hit]:
        n = max(k, self.candidates)
        return reciprocal_rank_fusion([r.retrieve(query, n) for r in self.retrievers], k)


# ---------- reranking ----------

Scorer = Callable[[str, list[str]], list[float]]


def cross_encoder_scorer(model_name: str, device: str | None = None) -> Scorer:
    """A cross-encoder reads query and passage together: slower than embeddings, but much
    more precise, so it's applied only to a short candidate list."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)

    def score(query: str, texts: list[str]) -> list[float]:
        pairs = [(query, t) for t in texts]
        return [float(x) for x in model.predict(pairs, show_progress_bar=False)]

    return score


class RerankRetriever:
    def __init__(self, base: Retriever, scorer: Scorer, candidates: int = 30):
        self.base = base
        self.scorer = scorer
        self.candidates = candidates

    def retrieve(self, query: str, k: int) -> list[Hit]:
        cands = self.base.retrieve(query, max(k, self.candidates))
        if not cands:
            return []
        scores = self.scorer(query, [h.chunk.text for h in cands])
        ranked = sorted(zip(cands, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [Hit(h.chunk, s) for h, s in ranked[:k]]


# ---------- indexing ----------


def index_chunks(
    chunks: list[Chunk], embedder: Embedder, store: VectorStore, batch: int = 256
) -> int:
    store.reset()
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        store.upsert(part, embedder.embed_documents([c.text for c in part]))
    return store.count()
