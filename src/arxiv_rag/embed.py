"""Embedding models behind one small interface, so they are easy to swap and to fake in tests."""

from __future__ import annotations

from typing import Protocol

# BGE v1.5 models retrieve better when queries (not documents) carry this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(self.query_prefix + text, normalize_embeddings=True)
        return vec.tolist()
