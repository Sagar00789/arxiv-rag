"""Shared fixtures. Tests run offline: no model downloads, no API keys."""

import hashlib
import math

import pytest

from arxiv_rag.chunking import Chunk
from arxiv_rag.retrieve import DenseRetriever, index_chunks
from arxiv_rag.store import VectorStore


class HashingEmbedder:
    """Deterministic bag-of-words embedder: shared words => similar vectors."""

    dim = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in text.lower().split():
            v[int(hashlib.md5(w.strip(".,?").encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


CORPUS = [
    Chunk("p1::0", "p1", "Dense Passage Retrieval", "dual encoder dense retrieval of passages", 0),
    Chunk("p1::1", "p1", "Dense Passage Retrieval", "in-batch negatives improve training", 1),
    Chunk("p2::0", "p2", "ReAct", "agents interleave reasoning traces and tool actions", 0),
    Chunk("p3::0", "p3", "LoRA", "low rank adapters for efficient fine tuning", 0),
]


@pytest.fixture
def retriever():
    emb = HashingEmbedder()
    store = VectorStore(":memory:", "test", emb.dim)
    index_chunks(CORPUS, emb, store)
    return DenseRetriever(emb, store)
