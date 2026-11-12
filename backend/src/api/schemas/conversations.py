from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str | None
    summary: str | None
    status: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime


class CreateTurnRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: UUID | None = None


class CitationResponse(BaseModel):
    citation_label: str
    document_id: str
    chunk_id: str
    page_number: int
    source: str
    title: str | None


class TurnResponse(BaseModel):
    run_id: str
    message_id: str
    answer: str
    route: str
    review_case_id: str | None
    citations: list[CitationResponse]
    sufficient_evidence: bool
    safety_notice: str
