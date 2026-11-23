import pytest

from arxiv_rag.config import Settings
from arxiv_rag.pipeline import build_retriever
from arxiv_rag.retrieve import (
    BM25Retriever,
    Hit,
    HybridRetriever,
    RerankRetriever,
    reciprocal_rank_fusion,
    tokenize,
)

from .conftest import CORPUS


def hits(*ids):
    by_id = {c.chunk_id: c for c in CORPUS}
    return [Hit(by_id[i], 0.0) for i in ids]


def test_tokenize_keeps_technical_terms():
    assert tokenize("How does MAIN-RAG beat GPT-4 by 3.5 points?") == [
        "main-rag", "beat", "gpt-4", "3.5", "points",
    ]  # fmt: skip


def test_bm25_matches_exact_keywords():
    top = BM25Retriever(CORPUS).retrieve("low rank adapters", k=2)
    assert top[0].chunk.chunk_id == "p3::0"


def test_bm25_returns_nothing_for_unknown_words():
    assert BM25Retriever(CORPUS).retrieve("zzz qqq", k=3) == []


def test_rrf_rewards_agreement_between_rankers():
    a = hits("p1::0", "p2::0")
    b = hits("p3::0", "p2::0")
    fused = [h.chunk.chunk_id for h in reciprocal_rank_fusion([a, b], k=3)]
    # p2::0 is only 2nd in each list, but found by both, so it beats each list's #1
    assert fused[0] == "p2::0"
    assert set(fused) == {"p1::0", "p2::0", "p3::0"}


def test_hybrid_combines_both_retrievers(retriever):
    hybrid = HybridRetriever([retriever, BM25Retriever(CORPUS)], candidates=4)
    top = hybrid.retrieve("agents tool actions", k=2)
    assert top[0].chunk.paper_id == "p2"
    assert len(top) == 2


def test_reranker_reorders_candidates(retriever):
    # A fake scorer that prefers the LoRA chunk regardless of the base ranking.
    def scorer(query, texts):
        return [1.0 if "low rank" in t else 0.0 for t in texts]

    top = RerankRetriever(retriever, scorer, candidates=4).retrieve("dense retrieval", k=1)
    assert top[0].chunk.chunk_id == "p3::0"


def test_unknown_retriever_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever(Settings(), "magic")
