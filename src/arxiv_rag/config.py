"""Settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def model_short_name(model: str) -> str:
    """'BAAI/bge-base-en-v1.5' -> 'bge-base-en-v1.5'"""
    return model.rstrip("/").split("/")[-1]


@dataclass(frozen=True)
class Settings:
    llm_model: str = field(
        default_factory=lambda: _env("LLM_MODEL", "gemini/gemini-3.5-flash-lite")
    )
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", DEFAULT_EMBED_MODEL))
    # sentence-transformers (PyTorch, used for all results) | fastembed (ONNX, used by the demo)
    embed_backend: str = field(
        default_factory=lambda: _env("EMBED_BACKEND", "sentence-transformers")
    )
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", "qdrant_data"))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "arxiv_chunks"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "300")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))
    # dense | bm25 | hybrid[@bm25_weight], optionally + "-rerank" (see README results)
    retriever: str = field(default_factory=lambda: _env("RETRIEVER", "hybrid@0.8"))
    reranker_model: str = field(
        default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))

    @property
    def dense_collection(self) -> str:
        """One Qdrant collection per embedding model, so several can be compared side by side.
        The default model keeps the plain name, so existing indexes keep working."""
        suffix = "__fastembed" if self.embed_backend == "fastembed" else ""
        if self.embed_model == DEFAULT_EMBED_MODEL:
            return self.collection + suffix
        slug = re.sub(r"[^a-z0-9]+", "-", model_short_name(self.embed_model).lower()).strip("-")
        return f"{self.collection}__{slug}{suffix}"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def papers_path(self) -> Path:
        return self.data_dir / "papers.jsonl"
