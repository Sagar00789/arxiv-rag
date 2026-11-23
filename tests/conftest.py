"""Shared fixtures. Tests run offline: no model downloads, no API keys."""

import hashlib
import math
import sys
import types

import pytest

from arxiv_rag import generate
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


# ---------- fake litellm (for LLM retry / quota tests) ----------


def fake_litellm(script):
    """A stand-in litellm whose completion() replays `script` (exceptions or reply text)."""
    mod = types.ModuleType("litellm")
    for name in (
        "NotFoundError",
        "BadRequestError",
        "AuthenticationError",
        "RateLimitError",
        "APIConnectionError",
    ):
        setattr(mod, name, type(name, (Exception,), {}))
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        step = script.pop(0)
        if isinstance(step, tuple):  # (exception name, message)
            raise getattr(mod, step[0])(step[1])
        msg = types.SimpleNamespace(content=step)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    mod.completion = completion
    mod.calls = calls
    return mod


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(generate.time, "sleep", waited.append)
    generate.set_rate_limit(0)
    return waited


def install(monkeypatch, script):
    mod = fake_litellm(script)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod
