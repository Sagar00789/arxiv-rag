"""Wires retrieval and generation together."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .generate import generate_answer
from .retrieve import DenseRetriever, Hit, Retriever


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
    def from_settings(cls, settings: Settings | None = None) -> RAGPipeline:
        from .embed import SentenceTransformerEmbedder
        from .store import VectorStore

        s = settings or Settings()
        embedder = SentenceTransformerEmbedder(s.embed_model)
        store = VectorStore(s.qdrant_path, s.collection, embedder.dim)
        return cls(DenseRetriever(embedder, store), s.llm_model, s.top_k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retriever.retrieve(question, k or self.top_k)
        return Answer(question, generate_answer(question, hits, self.llm_model), hits)
