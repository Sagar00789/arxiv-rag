"""Qdrant vector store. Runs embedded (a local folder), so there's no server to set up."""

from __future__ import annotations

import atexit
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Chunk

_clients: dict[str, QdrantClient] = {}


def get_client(path: str) -> QdrantClient:
    """One client per storage folder: embedded Qdrant locks its folder, so every collection
    in it must share a client."""
    if path == ":memory:":
        return QdrantClient(location=":memory:")
    if path not in _clients:
        _clients[path] = QdrantClient(path=path)
        # Close cleanly before interpreter shutdown (avoids a noisy warning from __del__).
        atexit.register(_clients[path].close)
    return _clients[path]


def _point_id(chunk_id: str) -> str:
    # Qdrant needs UUID or int ids; derive a stable UUID from our string id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, path: str, collection: str, dim: int):
        self.client = get_client(path)
        self.collection = collection
        self.dim = dim

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]], batch: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for i in range(0, len(chunks), batch):
            points = [
                PointStruct(id=_point_id(c.chunk_id), vector=v, payload=asdict(c))
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch], strict=True)
            ]
            self.client.upsert(self.collection, points=points)

    def search(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        res = self.client.query_points(self.collection, query=vector, limit=k, with_payload=True)
        return [(Chunk(**p.payload), p.score) for p in res.points]

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        return self.client.count(self.collection).count
