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
        # renamed in newer sentence-transformers; support both
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
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
        vec = self.model.encode(
            self.query_prefix + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vec.tolist()


class FastEmbedEmbedder:
    """Same BGE models via ONNX Runtime (FastEmbed): no PyTorch, ~10x less memory. Used by the
    hosted demo. Its weights are quantized, so its vectors differ slightly from the
    sentence-transformers version: never mix the two in one index."""

    def __init__(self, model_name: str, batch_size: int = 64):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)
        self.batch_size = batch_size
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""
        self.dim = len(self._encode(["probe"])[0])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for v in self.model.embed(texts, batch_size=self.batch_size):
            norm = float((v * v).sum()) ** 0.5 or 1.0
            out.append((v / norm).tolist())
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


EMBED_BACKENDS = ("sentence-transformers", "fastembed")


def make_embedder(model_name: str, backend: str = "sentence-transformers") -> Embedder:
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown EMBED_BACKEND {backend!r}; choose from {', '.join(EMBED_BACKENDS)}")
