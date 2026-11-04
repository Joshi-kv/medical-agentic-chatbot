import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from google.genai.errors import APIError
from langchain_core.embeddings import Embeddings
from langchain_google_genai._common import GoogleGenerativeAIError

from config.settings import get_settings
from ingestion.chunker import chunk_pages
from ingestion.normalizer import normalize_pages
from ingestion.pdf_loader import load_pdf
from ingestion.schemas import (
    DocumentChunk,
    IngestionResult,
    IngestionStatus,
    SourceDocument,
)
from integrations.google_genai import get_google_genai_embeddings
from integrations.pinecone import get_pinecone_index

EMBEDDING_BATCH_SIZE = 32
EMBEDDING_RATE_LIMIT_MAX_RETRIES = 5
EMBEDDING_RATE_LIMIT_DEFAULT_DELAY_SECONDS = 60.0
PINECONE_BATCH_SIZE = 100

Item = TypeVar("Item")


def ingest_pdf(document: SourceDocument, file_path: Path) -> IngestionResult:
    """
    Ingest a PDF document into the system.
    """
    started_at = datetime.now(timezone.utc)

    pages_processed = 0
    pages_requiring_ocr = 0
    chunks_created = 0
    vectors_upserted = 0

    try:
        pages = load_pdf(
            document=document,
            file_path=file_path,
        )

        pages_processed = len(pages)
        normalized_pages = normalize_pages(pages)

        pages_requiring_ocr = sum(page.requires_ocr for page in normalized_pages)

        chunks = chunk_pages(normalized_pages, document)

        chunks_created = len(chunks)

        if not chunks:
            raise ValueError(f"No chunks created from {document.document_id}")

        vectors = _embed_chunks(chunks)

        index = get_pinecone_index()
        settings = get_settings()

        attempted_ids: list[str] = []

        try:
            for vector_batch in _batched(vectors, PINECONE_BATCH_SIZE):
                attempted_ids.extend([vector["id"] for vector in vector_batch])

                index.upsert(
                    namespace=settings.pinecone_namespace, vectors=vector_batch
                )

                vectors_upserted += len(vector_batch)
        except Exception:
            _rollback_vectors(
                vector_ids=attempted_ids,
                namespace=settings.pinecone_namespace,
            )
            raise

        return IngestionResult(
            document_id=document.document_id,
            status=IngestionStatus.COMPLETED,
            pages_processed=pages_processed,
            pages_requiring_ocr=pages_requiring_ocr,
            chunks_created=chunks_created,
            vectors_upserted=vectors_upserted,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as error:
        return IngestionResult(
            document_id=document.document_id,
            status=IngestionStatus.FAILED,
            pages_processed=pages_processed,
            pages_requiring_ocr=pages_requiring_ocr,
            chunks_created=chunks_created,
            vectors_upserted=vectors_upserted,
            errors=[f"{type(error).__name__}: {error}"],
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )


def _embed_chunks(chunks: list[DocumentChunk]) -> list[dict]:
    """Embed chunks and prepare them for upserting."""

    embeddings_client = get_google_genai_embeddings()
    records: list[dict] = []

    for chunk_batch in _batched(chunks, EMBEDDING_BATCH_SIZE):
        texts = [chunk.text for chunk in chunk_batch]

        embeddings = _embed_documents_with_rate_limit_retry(
            embeddings_client,
            texts,
        )

        if len(embeddings) != len(chunk_batch):
            raise RuntimeError(
                "Embedding provider returned an unexpected vector count."
            )

        for chunk, embedding in zip(
            chunk_batch,
            embeddings,
            strict=True,
        ):
            records.append(
                _create_vector_record(
                    chunk=chunk,
                    embedding=embedding,
                )
            )

    return records


def _embed_documents_with_rate_limit_retry(
    embeddings_client: Embeddings,
    texts: list[str],
) -> list[list[float]]:
    for retry_count in range(EMBEDDING_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return embeddings_client.embed_documents(texts)
        except GoogleGenerativeAIError as error:
            delay = _google_rate_limit_retry_delay(error)
            if delay is None or retry_count == EMBEDDING_RATE_LIMIT_MAX_RETRIES:
                raise
            time.sleep(delay)

    raise RuntimeError("Embedding retry loop exited unexpectedly.")


def _google_rate_limit_retry_delay(
    error: GoogleGenerativeAIError,
) -> float | None:
    cause = error.__cause__
    if not isinstance(cause, APIError) or cause.code != 429:
        return None

    payload = cause.details
    if not isinstance(payload, dict):
        return EMBEDDING_RATE_LIMIT_DEFAULT_DELAY_SECONDS

    error_payload = payload.get("error", payload)
    if not isinstance(error_payload, dict):
        return EMBEDDING_RATE_LIMIT_DEFAULT_DELAY_SECONDS

    for detail in error_payload.get("details", []):
        if not isinstance(detail, dict):
            continue
        if not str(detail.get("@type", "")).endswith("RetryInfo"):
            continue

        retry_delay = detail.get("retryDelay")
        if isinstance(retry_delay, str) and retry_delay.endswith("s"):
            try:
                return max(float(retry_delay[:-1]), 0.0) + 1.0
            except ValueError:
                break

    return EMBEDDING_RATE_LIMIT_DEFAULT_DELAY_SECONDS


def _create_vector_record(
    chunk: DocumentChunk,
    embedding: list[float],
) -> dict:
    """
    Create a vector record from a chunk and its embedding.
    """
    metadata = {
        **chunk.metadata,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "page_number": chunk.page_number,
        "chunk_number": chunk.chunk_number,
        "content_hash": chunk.content_hash,
        "source": chunk.source,
        "tenant_id": chunk.tenant_id,
    }

    if chunk.title is not None:
        metadata["title"] = chunk.title

    return {
        "id": chunk.chunk_id,
        "values": embedding,
        "metadata": metadata,
    }


def _rollback_vectors(
    vector_ids: list[str],
    namespace: str,
) -> None:
    if not vector_ids:
        return

    try:
        get_pinecone_index().delete(
            namespace=namespace,
            ids=vector_ids,
        )
    except Exception:
        # Rollback failure should not hide the original ingestion error.
        pass


def _batched(
    items: list[Item],
    batch_size: int,
) -> Iterator[list[Item]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
