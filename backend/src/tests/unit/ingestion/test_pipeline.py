from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError
from langchain_google_genai._common import GoogleGenerativeAIError

from ingestion.exceptions import PDFNotFoundError
from ingestion.pipeline import (
    _batched,
    _create_vector_record,
    _embed_chunks,
    _rollback_vectors,
    ingest_pdf,
)
from ingestion.schemas import (
    DocumentChunk,
    ExtractedPage,
    IngestionStatus,
    SourceDocument,
)


def make_document(**updates: object) -> SourceDocument:
    values = {
        "document_id": "doc-001",
        "filename": "cardiology_notes.pdf",
        "size_bytes": 2048,
        "checksum": "c" * 64,
        "source": "cardiology_notes.pdf",
        "title": "Cardiology Notes",
        "tenant_id": "tenant-clinic",
        "metadata": {"department": "cardiology"},
    }
    values.update(updates)
    return SourceDocument(**values)


def make_chunk(**updates: object) -> DocumentChunk:
    values = {
        "chunk_id": "doc-001:page-1:chunk-0",
        "document_id": "doc-001",
        "page_number": 1,
        "chunk_number": 0,
        "text": "Echocardiogram shows normal ejection fraction.",
        "content_hash": "e" * 64,
        "source": "cardiology_notes.pdf",
        "title": "Cardiology Notes",
        "tenant_id": "tenant-clinic",
        "metadata": {"department": "cardiology"},
    }
    values.update(updates)
    return DocumentChunk(**values)


def test_batched_splits_items() -> None:
    items = list(range(10))
    batches = list(_batched(items, 3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


@pytest.mark.parametrize("invalid_size", [0, -1, -10])
def test_batched_rejects_non_positive_batch_size(invalid_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(_batched([1, 2, 3], invalid_size))


def test_create_vector_record_with_title() -> None:
    chunk = make_chunk(title="Medical Guide")
    embedding = [0.1, 0.2, 0.3]

    record = _create_vector_record(chunk, embedding)

    assert record["id"] == chunk.chunk_id
    assert record["values"] == embedding
    assert record["metadata"]["document_id"] == "doc-001"
    assert record["metadata"]["title"] == "Medical Guide"
    assert record["metadata"]["department"] == "cardiology"
    assert record["metadata"]["text"] == chunk.text
    assert record["metadata"]["page_number"] == 1
    assert record["metadata"]["chunk_number"] == 0
    assert record["metadata"]["content_hash"] == "e" * 64
    assert record["metadata"]["source"] == "cardiology_notes.pdf"
    assert record["metadata"]["tenant_id"] == "tenant-clinic"


def test_create_vector_record_without_title() -> None:
    chunk = make_chunk(title=None)
    embedding = [0.1, 0.2, 0.3]

    record = _create_vector_record(chunk, embedding)

    assert "title" not in record["metadata"]


def test_embed_chunks_success() -> None:
    chunks = [make_chunk(chunk_id=f"c-{i}") for i in range(3)]
    mock_embeddings_client = MagicMock()
    mock_embeddings_client.embed_documents.return_value = [
        [0.1 * i] * 3 for i in range(3)
    ]

    with patch(
        "ingestion.pipeline.get_google_genai_embeddings",
        return_value=mock_embeddings_client,
    ):
        records = _embed_chunks(chunks)

    assert len(records) == 3
    assert records[0]["id"] == "c-0"
    assert records[1]["id"] == "c-1"
    assert records[2]["id"] == "c-2"
    mock_embeddings_client.embed_documents.assert_called_once()


def test_embed_chunks_mismatched_count_raises_runtime_error() -> None:
    chunks = [make_chunk(chunk_id="c-0"), make_chunk(chunk_id="c-1")]
    mock_embeddings_client = MagicMock()
    # Returns only 1 vector for 2 chunks
    mock_embeddings_client.embed_documents.return_value = [[0.1, 0.2]]

    with patch(
        "ingestion.pipeline.get_google_genai_embeddings",
        return_value=mock_embeddings_client,
    ):
        with pytest.raises(RuntimeError, match="unexpected vector count"):
            _embed_chunks(chunks)


def test_embed_chunks_waits_and_retries_google_rate_limit() -> None:
    chunk = make_chunk(chunk_id="c-0")
    provider_error = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Embedding token quota exceeded.",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "2s",
                    }
                ],
            }
        },
    )
    rate_limit_error = GoogleGenerativeAIError("rate limited")
    rate_limit_error.__cause__ = provider_error

    embeddings_client = MagicMock()
    embeddings_client.embed_documents.side_effect = [
        rate_limit_error,
        [[0.1, 0.2, 0.3]],
    ]

    with (
        patch(
            "ingestion.pipeline.get_google_genai_embeddings",
            return_value=embeddings_client,
        ),
        patch("time.sleep") as sleep,
    ):
        records = _embed_chunks([chunk])

    assert records[0]["id"] == "c-0"
    assert records[0]["values"] == [0.1, 0.2, 0.3]
    sleep.assert_called_once_with(3.0)


def test_rollback_vectors_invokes_index_delete() -> None:
    mock_index = MagicMock()
    with patch("ingestion.pipeline.get_pinecone_index", return_value=mock_index):
        _rollback_vectors(["id-1", "id-2"], namespace="test-ns")

    mock_index.delete.assert_called_once_with(namespace="test-ns", ids=["id-1", "id-2"])


def test_rollback_vectors_no_op_on_empty_ids() -> None:
    mock_index = MagicMock()
    with patch("ingestion.pipeline.get_pinecone_index", return_value=mock_index):
        _rollback_vectors([], namespace="test-ns")

    mock_index.delete.assert_not_called()


def test_rollback_vectors_suppresses_exceptions() -> None:
    mock_index = MagicMock()
    mock_index.delete.side_effect = RuntimeError("Pinecone delete network error")
    with patch("ingestion.pipeline.get_pinecone_index", return_value=mock_index):
        # Should not raise
        _rollback_vectors(["id-1"], namespace="test-ns")


def test_ingest_pdf_success() -> None:
    doc = make_document()
    file_path = Path("/mock/cardiology.pdf")

    mock_pages = [
        ExtractedPage(
            document_id=doc.document_id,
            page_number=1,
            text="Page 1 text about cardiac rhythm.",
            requires_ocr=False,
        ),
        ExtractedPage(
            document_id=doc.document_id,
            page_number=2,
            text="Page 2 text about medications.",
            requires_ocr=True,
        ),
    ]

    mock_chunks = [
        make_chunk(chunk_id="c-1", page_number=1, text="Chunk 1"),
        make_chunk(chunk_id="c-2", page_number=2, text="Chunk 2"),
    ]

    mock_embeddings_client = MagicMock()
    mock_embeddings_client.embed_documents.return_value = [
        [0.1] * 768,
        [0.2] * 768,
    ]

    mock_index = MagicMock()
    mock_settings = MagicMock()
    mock_settings.pinecone_namespace = "medical-records"

    with (
        patch("ingestion.pipeline.load_pdf", return_value=mock_pages),
        patch("ingestion.pipeline.chunk_pages", return_value=mock_chunks),
        patch(
            "ingestion.pipeline.get_google_genai_embeddings",
            return_value=mock_embeddings_client,
        ),
        patch("ingestion.pipeline.get_pinecone_index", return_value=mock_index),
        patch("ingestion.pipeline.get_settings", return_value=mock_settings),
    ):
        result = ingest_pdf(doc, file_path)

    assert result.status == IngestionStatus.COMPLETED
    assert result.document_id == doc.document_id
    assert result.pages_processed == 2
    assert result.pages_requiring_ocr == 1
    assert result.chunks_created == 2
    assert result.vectors_upserted == 2
    assert result.errors == []
    assert result.completed_at is not None
    assert result.completed_at >= result.started_at

    mock_index.upsert.assert_called_once()
    upsert_call = mock_index.upsert.call_args
    assert upsert_call.kwargs["namespace"] == "medical-records"
    assert len(upsert_call.kwargs["vectors"]) == 2


def test_ingest_pdf_empty_chunks_returns_failed_result() -> None:
    doc = make_document()
    file_path = Path("/mock/empty.pdf")

    mock_pages = [
        ExtractedPage(
            document_id=doc.document_id,
            page_number=1,
            text="",
            requires_ocr=True,
        )
    ]

    with (
        patch("ingestion.pipeline.load_pdf", return_value=mock_pages),
        patch("ingestion.pipeline.chunk_pages", return_value=[]),
    ):
        result = ingest_pdf(doc, file_path)

    assert result.status == IngestionStatus.FAILED
    assert result.pages_processed == 1
    assert result.pages_requiring_ocr == 1
    assert result.chunks_created == 0
    assert result.vectors_upserted == 0
    assert len(result.errors) == 1
    assert "ValueError: No chunks created" in result.errors[0]


def test_ingest_pdf_load_pdf_error_returns_failed_result() -> None:
    doc = make_document()
    file_path = Path("/mock/missing.pdf")

    with patch(
        "ingestion.pipeline.load_pdf",
        side_effect=PDFNotFoundError(path=str(file_path)),
    ):
        result = ingest_pdf(doc, file_path)

    assert result.status == IngestionStatus.FAILED
    assert result.pages_processed == 0
    assert result.chunks_created == 0
    assert result.vectors_upserted == 0
    assert len(result.errors) == 1
    assert "PDFNotFoundError" in result.errors[0]


def test_ingest_pdf_upsert_failure_triggers_rollback() -> None:
    doc = make_document()
    file_path = Path("/mock/test.pdf")

    mock_pages = [
        ExtractedPage(
            document_id=doc.document_id,
            page_number=1,
            text="Valid page text",
            requires_ocr=False,
        )
    ]
    mock_chunks = [make_chunk(chunk_id="chunk-to-rollback")]

    mock_embeddings_client = MagicMock()
    mock_embeddings_client.embed_documents.return_value = [[0.1] * 768]

    mock_index = MagicMock()
    mock_index.upsert.side_effect = RuntimeError("Pinecone connection lost")

    mock_settings = MagicMock()
    mock_settings.pinecone_namespace = "medical-records"

    with (
        patch("ingestion.pipeline.load_pdf", return_value=mock_pages),
        patch("ingestion.pipeline.chunk_pages", return_value=mock_chunks),
        patch(
            "ingestion.pipeline.get_google_genai_embeddings",
            return_value=mock_embeddings_client,
        ),
        patch("ingestion.pipeline.get_pinecone_index", return_value=mock_index),
        patch("ingestion.pipeline.get_settings", return_value=mock_settings),
    ):
        result = ingest_pdf(doc, file_path)

    assert result.status == IngestionStatus.FAILED
    assert result.vectors_upserted == 0
    assert len(result.errors) == 1
    assert "RuntimeError: Pinecone connection lost" in result.errors[0]

    # Verify rollback was attempted with the chunk id
    mock_index.delete.assert_called_once_with(
        namespace="medical-records", ids=["chunk-to-rollback"]
    )
