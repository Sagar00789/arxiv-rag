"""Turn raw paper text into overlapping chunks.

Baseline strategy: fixed-size word windows with overlap. This is deliberately simple;
section-aware chunking is one of the experiments we measure against it later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

_REFERENCES = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    title: str
    text: str
    position: int


def clean_text(text: str) -> str:
    """Drop the reference list, re-join hyphenated line breaks, collapse whitespace."""
    matches = list(_REFERENCES.finditer(text))
    if matches and matches[-1].start() > len(text) * 0.5:
        text = text[: matches[-1].start()]
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into windows of `size` words, each sharing `overlap` words with the last."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_paper(
    paper_id: str, title: str, text: str, size: int = 300, overlap: int = 50
) -> list[Chunk]:
    pieces = chunk_text(clean_text(text), size=size, overlap=overlap)
    return [
        Chunk(chunk_id=f"{paper_id}::{i}", paper_id=paper_id, title=title, text=p, position=i)
        for i, p in enumerate(pieces)
    ]


def save_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
