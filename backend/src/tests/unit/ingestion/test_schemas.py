from datetime import datetime

import pytest
from pydantic import ValidationError

from ingestion.schemas import (
    DocumentChunk,
    ExtractedPage,
    IngestionResult,
    IngestionStatus,
    SourceDocument,
)


def test_source_document_validation() -> None:
    valid_doc = SourceDocument(
        document_id="doc-1",
        filename="report.pdf",
        source="ehr",
        size_bytes=5000,
        checksum="0" * 64,
    )
    assert valid_doc.document_id == "doc-1"
    assert valid_doc.content_type == "application/pdf"
    assert valid_doc.tenant_id == "default"
    assert isinstance(valid_doc.created_at, datetime)

    # Invalid checksum (not 64 characters hex)
    with pytest.raises(ValidationError):
        SourceDocument(
            document_id="doc-1",
            filename="report.pdf",
            source="ehr",
            size_bytes=5000,
            checksum="short",
        )

    # Invalid size_bytes (must be >= 1)
    with pytest.raises(ValidationError):
        SourceDocument(
            document_id="doc-1",
            filename="report.pdf",
            source="ehr",
            size_bytes=0,
            checksum="0" * 64,
        )

    # Extra fields forbidden
    with pytest.raises(ValidationError):
        SourceDocument(
            document_id="doc-1",
            filename="report.pdf",
            source="ehr",
            size_bytes=100,
            checksum="0" * 64,
            unexpected_field="disallowed",  # type: ignore[call-arg]
        )


def test_extracted_page_validation() -> None:
    page = ExtractedPage(
        document_id="doc-1",
        page_number=1,
        text="Normal sinus rhythm.",
        requires_ocr=False,
    )
    assert page.page_number == 1
    assert page.text == "Normal sinus rhythm."

    # page_number must be >= 1
    with pytest.raises(ValidationError):
        ExtractedPage(
            document_id="doc-1",
            page_number=0,
            text="text",
        )


def test_document_chunk_validation() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        page_number=1,
        chunk_number=0,
        text="Sample chunk text",
        content_hash="f" * 64,
        source="ehr",
    )
    assert chunk.chunk_id == "chunk-1"
    assert chunk.content_hash == "f" * 64

    # Empty text disallowed
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            page_number=1,
            chunk_number=0,
            text="",
            content_hash="f" * 64,
            source="ehr",
        )


def test_ingestion_result_validation() -> None:
    result = IngestionResult(
        document_id="doc-1",
        status=IngestionStatus.COMPLETED,
        pages_processed=5,
        chunks_created=12,
        vectors_upserted=12,
    )
    assert result.status == IngestionStatus.COMPLETED
    assert result.pages_processed == 5
    assert result.errors == []
