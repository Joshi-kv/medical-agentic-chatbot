from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: UUID | None = None


class RetrievalMatchResponse(BaseModel):
    chunk_id: str
    score: float
    text: str
    document_id: str
    page_number: int
    chunk_number: int
    source: str
    title: str | None = None


class RetrievalResponse(BaseModel):
    query: str
    matches: list[RetrievalMatchResponse]
