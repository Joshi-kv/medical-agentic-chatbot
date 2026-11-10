from uuid import UUID

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: UUID | None = None


class CitationResponse(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    page_number: int
    source: str
    title: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    sufficient_evidence: bool
    safety_notice: str
