"""Multiple embedding models, answer-quality eval, exact-ID ingest and the web UI helpers."""

import dataclasses
import json

import pytest
from typer.testing import CliRunner

from arxiv_rag import pipeline, ui
from arxiv_rag.answer_eval import _ratio, evaluate_answers, summarize
from arxiv_rag.chunking import save_chunks
from arxiv_rag.cli import app
from arxiv_rag.config import DEFAULT_EMBED_MODEL, Settings
from arxiv_rag.evaluation import TestItem, save_testset
from arxiv_rag.generate import LLMError
from arxiv_rag.ingest import read_id_list
from arxiv_rag.pipeline import RAGPipeline
from arxiv_rag.store import get_client

from .conftest import CORPUS, install

# ---------- embedding models ----------


def test_each_embedding_model_gets_its_own_collection():
    s = Settings()
    assert dataclasses.replace(s, embed_model=DEFAULT_EMBED_MODEL).dense_collection == s.collection
    base = dataclasses.replace(s, embed_model="BAAI/bge-base-en-v1.5").dense_collection
    large = dataclasses.replace(s, embed_model="BAAI/bge-large-en-v1.5").dense_collection
    assert base == f"{s.collection}__bge-base-en-v1-5"
    assert len({s.collection, base, large}) == 3


def test_one_qdrant_client_per_folder(tmp_path):
    # Embedded Qdrant locks its folder, so two collections must share one client.
    assert get_client(str(tmp_path)) is get_client(str(tmp_path))


def test_eval_compares_embedding_models(tmp_path, monkeypatch, retriever):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    save_chunks(CORPUS, tmp_path / "data" / "chunks.jsonl")
    save_testset([TestItem("low rank adapters", "p3", "p3::0")], tmp_path / "eval/testset.jsonl")
    seen = []

    def fake_dense(model, path, collection, backend="sentence-transformers"):
        seen.append((model, collection))
        return retriever

    monkeypatch.setattr(pipeline, "_dense", fake_dense)
    args = ["eval", "-r", "dense", "-r", "bm25", "-e", DEFAULT_EMBED_MODEL]
    args += ["-e", "BAAI/bge-base-en-v1.5"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    rows = (tmp_path / "eval" / "RESULTS.md").read_text()
    assert "| dense |" in rows
    assert "| dense · bge-base-en-v1.5 |" in rows
    assert rows.count("| bm25 |") == 1  # BM25 doesn't depend on the embedding model
    assert {c for _, c in seen} == {"arxiv_chunks", "arxiv_chunks__bge-base-en-v1-5"}


def test_missing_index_is_a_clear_error(tmp_path, monkeypatch):
    def missing(*a):
        raise pipeline.IndexMissingError("No index for embedding model X. Run: arxiv-rag index")

    monkeypatch.setattr(pipeline, "_dense", missing)
    t = tmp_path / "testset.jsonl"
    save_testset([TestItem("q", "p1", "p1::0")], t)
    result = CliRunner().invoke(app, ["eval", "-r", "dense", "--testset", str(t)])
    assert result.exit_code == 1
    assert "Run: arxiv-rag index" in result.output
    assert "Traceback" not in result.output


# ---------- answer-quality eval ----------


def judge(supported, total, cites_ok, cites, correctness, refused=False):
    return json.dumps(
        {
            "supported_claims": supported,
            "total_claims": total,
            "correct_citations": cites_ok,
            "total_citations": cites,
            "correctness": correctness,
            "refused": refused,
            "note": "ok",
        }
    )


def test_ratio_handles_bad_judge_output():
    assert _ratio(3, 4) == 0.75
    assert _ratio(5, 4) == 1.0  # judge over-counted; cap at 1
    assert _ratio(0, 0) is None
    assert _ratio("x", 2) is None


def test_answer_eval_scores_and_resumes(monkeypatch, sleeps, tmp_path, retriever):
    items = [
        TestItem("low rank adapters fine tuning", "p3", "p3::0", "LoRA"),
        TestItem("agents tool actions", "p2", "p2::0", "ReAct"),
        TestItem("zzz unrelated", "p1", "p1::1", "?"),
    ]
    log = tmp_path / "answers.jsonl"
    # Q1: answered well. Q2: then the daily quota runs out mid-run.
    install(monkeypatch, ["LoRA adds low rank adapters [1].", judge(1, 1, 1, 1, "yes"),
                          ("RateLimitError", "PerDay")])  # fmt: skip
    with pytest.raises(LLMError):
        evaluate_answers(retriever, items, "m", "judge", log, k=2)
    assert len(log.read_text().splitlines()) == 1  # Q1 kept

    install(monkeypatch, ["ReAct interleaves reasoning [1] and acting [2].",
                          judge(1, 2, 1, 2, "partial"),
                          "I couldn't find this in the indexed papers.",
                          judge(0, 0, 0, 0, "no", refused=True)])  # fmt: skip
    judged = evaluate_answers(retriever, items, "m", "judge", log, k=2)
    assert [j.question for j in judged] == [it.question for it in items]

    s = summarize(judged)
    assert s["n"] == 3
    assert s["faithfulness"] == pytest.approx((1.0 + 0.5) / 2)  # refusal excluded
    assert s["citation_accuracy"] == pytest.approx((1.0 + 0.5) / 2)
    assert s["correctness"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-3)
    assert s["refusal_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert judged[0].gold_retrieved is True


# ---------- reproducible corpus ----------


def test_read_id_list(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# header\n2410.16285v1\n\n2603.18330v1  # comment\n")
    assert read_id_list(f) == ["2410.16285v1", "2603.18330v1"]


# ---------- web UI ----------


@pytest.fixture
def fake_pipe(retriever, monkeypatch):
    monkeypatch.setattr(pipeline, "generate_answer", lambda q, h, m: "Answer [1].")
    return RAGPipeline(retriever, "fake")


def test_ui_answer_and_sources(fake_pipe):
    answer, sources = ui.answer_question("agents tool actions", "dense", lambda r: fake_pipe)
    assert answer.startswith("Answer [1].")
    assert "https://arxiv.org/abs/p2" in sources


def test_ui_shows_llm_errors_as_messages(fake_pipe, monkeypatch):
    def boom(q, h, m):
        raise LLMError("Daily free-tier quota is used up.")

    monkeypatch.setattr(pipeline, "generate_answer", boom)
    answer, _ = ui.answer_question("agents tool actions", "dense", lambda r: fake_pipe)
    assert "quota" in answer and answer.startswith("⚠️")


def test_ui_rejects_empty_question(fake_pipe):
    assert ui.answer_question("  ", "dense", lambda r: fake_pipe)[0] == "Please type a question."


def test_ui_results_tab_reads_readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# x\n## Results\n| a | b |\n## Quickstart\n")
    assert ui.results_markdown(readme).startswith("## Results")
    assert "GitHub" in ui.results_markdown(tmp_path / "missing.md")


def test_answer_results_record_which_models(tmp_path):
    from arxiv_rag.answer_eval import record_answer_results

    keys = ["faithfulness", "citation_accuracy", "correctness", "gold_retrieved"]
    summary = {"n": 1, "refusal_rate": 0.0, "correctness_when_gold_retrieved": 1.0}
    summary |= dict.fromkeys(keys, 1.0)
    record_answer_results(summary, "x", tmp_path, "gemini/a-lite", "gemini/b-judge")
    row = (tmp_path / "ANSWERS.md").read_text().splitlines()[-1]
    assert "| a-lite | b-judge |" in row
