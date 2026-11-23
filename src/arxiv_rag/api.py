"""HTTP API. Run with: uvicorn arxiv_rag.api:app --reload"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

app = FastAPI(title="arxiv-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    paper_id: str
    title: str
    chunk_id: str
    score: float
    url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline.from_settings()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    ans = get_pipeline().ask(req.question, req.k)
    return AskResponse(
        answer=ans.answer,
        sources=[
            Source(
                paper_id=h.chunk.paper_id,
                title=h.chunk.title,
                chunk_id=h.chunk.chunk_id,
                score=round(h.score, 4),
                url=f"https://arxiv.org/abs/{h.chunk.paper_id}",
            )
            for h in ans.sources
        ],
    )
