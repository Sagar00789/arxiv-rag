"""Weighted fusion, retriever names, paraphrasing and word overlap."""

import json

import pytest
from typer.testing import CliRunner

from arxiv_rag.chunking import save_chunks
from arxiv_rag.cli import app
from arxiv_rag.evaluation import (
    TestItem,
    load_testset,
    mean_overlap,
    paraphrase_testset,
    save_testset,
    word_overlap,
)
from arxiv_rag.generate import LLMError
from arxiv_rag.pipeline import parse_retriever
from arxiv_rag.retrieve import reciprocal_rank_fusion

from .conftest import CORPUS, install
from .test_hybrid import hits


@pytest.mark.parametrize(
    "name,expected",
    [
        ("dense", ("dense", 0.5, False)),
        ("bm25-rerank", ("bm25", 0.5, True)),
        ("hybrid@0.8", ("hybrid", 0.8, False)),
        ("hybrid@0.8-rerank", ("hybrid", 0.8, True)),
    ],
)
def test_parse_retriever(name, expected):
    assert parse_retriever(name) == expected


@pytest.mark.parametrize("bad", ["bm25@0.5", "hybrid@2", "hybrid-dense", "magic"])
def test_parse_retriever_rejects(bad):
    with pytest.raises(ValueError):
        parse_retriever(bad)


def test_weights_shift_the_fused_ranking():
    dense = hits("p1::0", "p2::0")
    bm25 = hits("p3::0", "p1::1")
    equal = reciprocal_rank_fusion([dense, bm25], k=1)
    bm25_heavy = reciprocal_rank_fusion([dense, bm25], k=1, weights=[0.2, 0.8])
    assert equal[0].chunk.chunk_id == "p1::0"  # tie on rank 1 -> first list wins
    assert bm25_heavy[0].chunk.chunk_id == "p3::0"


def test_rrf_weight_count_must_match():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([hits("p1::0")], k=1, weights=[0.5, 0.5])


def test_word_overlap():
    passage = "MAIN-RAG filters noisy documents with multiple agents"
    # main-rag, noisy, documents match; "filter" doesn't match "filters" (no stemming)
    assert word_overlap("How does MAIN-RAG filter noisy documents?", passage) == 0.75
    assert word_overlap("", passage) == 0.0


def test_paraphrase_keeps_gold_labels_and_resumes(monkeypatch, sleeps, tmp_path):
    chunks = {c.chunk_id: c for c in CORPUS}
    items = [TestItem("low rank adapters?", "p3", "p3::0"), TestItem("q2", "p2", "p2::0")]
    out = tmp_path / "para.jsonl"

    first = json.dumps({"question": "cheap ways to tune big models?"})
    install(monkeypatch, [first, ("RateLimitError", "PerDay")])
    with pytest.raises(LLMError):
        paraphrase_testset(items, chunks, "m", out)
    assert len(load_testset(out)) == 1

    install(monkeypatch, [json.dumps({"question": "how do agents use tools?"})])
    done = paraphrase_testset(items, chunks, "m", out)
    assert [d.gold_chunk_id for d in done] == ["p3::0", "p2::0"]
    assert done[0].question == "cheap ways to tune big models?"
    assert mean_overlap(done, chunks) < mean_overlap(items, chunks)


def test_eval_labels_rows_by_test_set(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_chunks(CORPUS, tmp_path / "data" / "chunks.jsonl")
    save_testset(
        [TestItem("low rank adapters", "p3", "p3::0")], tmp_path / "eval" / "testset_para.jsonl"
    )
    # make sure Settings picks up tmp paths
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    result = CliRunner().invoke(app, ["eval", "-r", "bm25", "--testset", "eval/testset_para.jsonl"])
    assert result.exit_code == 0, result.output
    assert "| bm25 [para] |" in (tmp_path / "eval" / "RESULTS.md").read_text()


def test_eval_rejects_bad_retriever_before_loading_anything(tmp_path):
    t = tmp_path / "testset.jsonl"
    save_testset([TestItem("q", "p1", "p1::0")], t)
    result = CliRunner().invoke(app, ["eval", "-r", "magic", "--testset", str(t)])
    assert result.exit_code == 1
    assert "Unknown retriever" in result.output
