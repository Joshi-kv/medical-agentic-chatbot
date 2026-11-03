from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MetadataValue = str | int | float | bool | list[str]


class IngestionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceDocument(BaseModel):
    """Information about the original uploaded document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    content_type: Literal["application/pdf"] = "application/pdf"
    size_bytes: int = Field(ge=1)

    # SHA-256 hash of the original file.
    checksum: str = Field(pattern=r"^[a-f0-9]{64}$")

    title: str | None = None
    source: str
    tenant_id: str = "default"

    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExtractedPage(BaseModel):
    """Text extracted from one PDF page."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    page_number: int = Field(ge=1)
    text: str

    # True when the page has images but little or no extractable text.
    requires_ocr: bool = False

    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """One searchable unit sent for embedding and storage."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str

    page_number: int = Field(ge=1)
    chunk_number: int = Field(ge=0)

    text: str = Field(min_length=1)

    # Hash of normalized chunk text.
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    source: str
    title: str | None = None
    tenant_id: str = "default"

    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class IngestionResult(BaseModel):
    """Summary returned by the complete ingestion pipeline."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    status: IngestionStatus

    pages_processed: int = Field(default=0, ge=0)
    pages_requiring_ocr: int = Field(default=0, ge=0)
    chunks_created: int = Field(default=0, ge=0)
    vectors_upserted: int = Field(default=0, ge=0)

    errors: list[str] = Field(default_factory=list)

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
