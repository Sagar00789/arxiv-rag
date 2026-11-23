from arxiv_rag.evaluation import (
    TestItem,
    _parse_json,
    evaluate_retrieval,
    hit_at_k,
    load_testset,
    reciprocal_rank,
    record_results,
    save_testset,
)
from arxiv_rag.generate import build_messages


def test_metrics():
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5
    assert reciprocal_rank(["a"], "z") == 0.0
    assert hit_at_k(["a", "b", "c"], "c", 2) == 0.0
    assert hit_at_k(["a", "b", "c"], "c", 3) == 1.0


def test_retriever_finds_relevant_chunk(retriever):
    hits = retriever.retrieve("how do agents use tool actions", k=2)
    assert hits[0].chunk.paper_id == "p2"


def test_evaluate_retrieval_scores(retriever):
    items = [
        TestItem("low rank adapters fine tuning", "p3", "p3::0"),
        TestItem("dense retrieval dual encoder", "p1", "p1::0"),
    ]
    res = evaluate_retrieval(retriever, items, k=3)
    assert res["n"] == 2
    assert res["chunk_hit@3"] == 1.0
    assert res["paper_mrr"] == 1.0


def test_testset_roundtrip_and_results_log(tmp_path):
    items = [TestItem("q?", "p1", "p1::0", "a")]
    save_testset(items, tmp_path / "t.jsonl")
    assert load_testset(tmp_path / "t.jsonl") == items

    res = {
        "n": 1,
        "k": 5,
        "chunk_hit@5": 1.0,
        "chunk_mrr": 1.0,
        "paper_hit@5": 1.0,
        "paper_mrr": 1.0,
        "p50_latency_ms": 3.2,
    }
    record_results(res, "exp-a", tmp_path)
    record_results(res, "exp-b", tmp_path)
    md = (tmp_path / "RESULTS.md").read_text()
    assert md.count("| exp-") == 2


def test_parse_json_tolerates_wrapping():
    assert _parse_json('```json\n{"question": "q", "answer": "a"}\n```') == {
        "question": "q",
        "answer": "a",
    }
    assert _parse_json("no json here") is None


def test_prompt_numbers_sources(retriever):
    hits = retriever.retrieve("dense retrieval", k=2)
    user = build_messages("What is DPR?", hits)[1]["content"]
    assert "[1]" in user and "[2]" in user and "What is DPR?" in user
