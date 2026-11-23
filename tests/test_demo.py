"""The hosted demo: FastEmbed backend and the in-memory bootstrap."""

import dataclasses
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from arxiv_rag import embed, pipeline, ui
from arxiv_rag.chunking import save_chunks
from arxiv_rag.config import Settings
from arxiv_rag.demo import LockedRetriever, build_demo
from arxiv_rag.embed import BGE_QUERY_PREFIX, make_embedder

from .conftest import CORPUS, HashingEmbedder


def test_fastembed_index_never_mixes_with_pytorch_index():
    s = Settings()
    fe = dataclasses.replace(s, embed_backend="fastembed")
    assert fe.dense_collection == s.dense_collection + "__fastembed"
    big = dataclasses.replace(fe, embed_model="BAAI/bge-base-en-v1.5")
    assert big.dense_collection.endswith("bge-base-en-v1-5__fastembed")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="EMBED_BACKEND"):
        make_embedder("m", "magic")


def test_fastembed_embedder_normalizes_and_prefixes_queries(monkeypatch):
    seen = []

    class FakeTextEmbedding:
        def __init__(self, name):
            self.name = name

        def embed(self, texts, batch_size=64):
            seen.extend(texts)
            for t in texts:
                yield np.array([3.0, 4.0]) * (len(t) % 3 + 1)

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    )
    e = embed.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    assert e.dim == 2
    assert e.embed_documents(["a doc"]) == [pytest.approx([0.6, 0.8])]  # unit length
    e.embed_query("what is rrf?")
    assert seen[-1] == BGE_QUERY_PREFIX + "what is rrf?"  # BGE query instruction added


@pytest.fixture
def demo(tmp_path, monkeypatch):
    s = dataclasses.replace(Settings(), data_dir=tmp_path, embed_backend="fastembed")
    save_chunks(CORPUS, s.chunks_path)  # corpus already present -> no arXiv download
    messages = []
    d = build_demo(s, tmp_path / "unused.txt", messages.append, lambda m: HashingEmbedder())
    monkeypatch.setattr(pipeline, "generate_answer", lambda q, h, m: "Answer [1].")
    return d, messages


def test_build_demo_indexes_in_memory(demo):
    d, messages = demo
    assert len(d.chunks) == len(CORPUS)
    assert any("Embedding" in m for m in messages)
    assert not any("Downloading" in m for m in messages)


@pytest.mark.parametrize("name", ["dense", "bm25", "hybrid", "hybrid@0.8"])
def test_demo_serves_every_listed_retriever(demo, name):
    d, _ = demo
    ans = d.pipeline(name).ask("low rank adapters fine tuning")
    assert ans.sources and ans.answer == "Answer [1]."
    assert {v for _, v in ui.RETRIEVER_CHOICES} <= {"dense", "bm25", "hybrid", "hybrid@0.8"}


def test_demo_pipelines_are_cached(demo):
    d, _ = demo
    assert d.pipeline("bm25") is d.pipeline("bm25")


def test_demo_has_no_reranker(demo):
    d, _ = demo
    with pytest.raises(ValueError, match="hosted demo"):
        d.pipeline("hybrid@0.8-rerank")


def test_ui_end_to_end_on_demo(demo):
    d, _ = demo
    answer, sources = ui.answer_question("agents tool actions", "hybrid@0.8", d.pipeline)
    assert answer.startswith("Answer [1].")
    assert "arxiv.org/abs/p2" in sources


def test_locked_retriever_serialises_access(retriever):
    lock = threading.Lock()
    r = LockedRetriever(retriever, lock)
    assert r.retrieve("dense retrieval", 1)
    assert not lock.locked()


def test_ui_shows_timing_split_and_score_meaning(demo):
    d, _ = demo
    answer, sources = ui.answer_question("agents tool actions", "hybrid@0.8", d.pipeline)
    assert "retrieval" in answer and "LLM" in answer
    assert "fusion score" in sources


@pytest.mark.parametrize(
    "name,label",
    [("dense", "cosine"), ("bm25", "BM25"), ("hybrid@0.8", "fusion"), ("bm25-rerank", "rerank")],
)
def test_score_label(name, label):
    assert label in ui.score_label(name)


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return super().embed_documents(texts)


def test_restart_reuses_cached_embeddings(tmp_path):
    s = dataclasses.replace(Settings(), data_dir=tmp_path, embed_backend="fastembed")
    save_chunks(CORPUS, s.chunks_path)
    first, second = CountingEmbedder(), CountingEmbedder()
    msgs = []
    build_demo(s, tmp_path / "x", msgs.append, lambda m: first)
    d = build_demo(s, tmp_path / "x", msgs.append, lambda m: second)
    assert first.calls == 1 and second.calls == 0  # second start skipped embedding
    assert any("cached embeddings" in m for m in msgs)
    assert (
        d.pipeline("dense").retriever.retrieve("low rank adapters", 1)[0].chunk.chunk_id == "p3::0"
    )


def test_cache_ignored_when_corpus_changes(tmp_path):
    s = dataclasses.replace(Settings(), data_dir=tmp_path, embed_backend="fastembed")
    save_chunks(CORPUS, s.chunks_path)
    build_demo(s, tmp_path / "x", embedder_factory=lambda m: HashingEmbedder())
    save_chunks(CORPUS[:2], s.chunks_path)  # different corpus
    again = CountingEmbedder()
    build_demo(s, tmp_path / "x", embedder_factory=lambda m: again)
    assert again.calls == 1


# ---------- memory-saving pieces ----------


def test_array_store_exact_cosine_search():
    from arxiv_rag.demo import ArrayStore

    emb = HashingEmbedder()
    store = ArrayStore(CORPUS, emb.embed_documents([c.text for c in CORPUS]))
    hits = store.search(emb.embed_query("low rank adapters"), 2)
    assert hits[0][0].chunk_id == "p3::0"
    assert hits[0][1] >= hits[1][1]  # sorted by score
    assert len(store.search(emb.embed_query("x"), 99)) == len(CORPUS)  # k capped


def test_array_store_rejects_mismatch():
    from arxiv_rag.demo import ArrayStore

    with pytest.raises(ValueError):
        ArrayStore(CORPUS, [[1.0, 0.0]])


def test_build_subprocess_streams_progress():
    from arxiv_rag.demo import run_build_subprocess

    lines = []
    cmd = [sys.executable, "-c", "print('Downloading...'); print('Build finished.')"]
    run_build_subprocess(Path("ids.txt"), lines.append, cmd=cmd)
    assert lines == ["Downloading...", "Build finished."]


def test_build_subprocess_failure_shows_its_output():
    from arxiv_rag.demo import run_build_subprocess

    cmd = [sys.executable, "-c", "print('step 1'); raise SystemExit('arXiv unreachable')"]
    with pytest.raises(RuntimeError, match="arXiv unreachable"):
        run_build_subprocess(Path("ids.txt"), lambda m: None, cmd=cmd)


def test_demo_never_loads_heavy_libraries():
    # The web process must stay small: no Qdrant client, PyTorch, PyMuPDF or LiteLLM at import.
    code = (
        "import sys, arxiv_rag.demo, arxiv_rag.ui; "
        "heavy = ('qdrant_client', 'torch', 'pymupdf', 'fitz', 'litellm'); "
        "print([m for m in heavy if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_fastembed_uses_small_batches():
    import inspect

    assert inspect.signature(embed.FastEmbedEmbedder).parameters["batch_size"].default <= 8
