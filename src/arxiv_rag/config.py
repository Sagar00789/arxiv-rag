"""Settings, read from environment variables (and a local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini/gemini-2.5-flash"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", "qdrant_data"))
    collection: str = field(default_factory=lambda: _env("COLLECTION", "arxiv_chunks"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "300")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def papers_path(self) -> Path:
        return self.data_dir / "papers.jsonl"
