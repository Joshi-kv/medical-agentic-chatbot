from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services.evidence import EvidenceService
from services.retrieval import RetrievalMatch, RetrievalResult


def make_match(
    chunk_id: str,
    score: float,
    content_hash: str | None = None,
) -> RetrievalMatch:
    return RetrievalMatch(
        chunk_id=chunk_id,
        score=score,
        text=f"Evidence from {chunk_id}",
        document_id="document-1",
        page_number=12,
        chunk_number=1,
        source="medical-book.pdf",
        title="Medical Book",
        content_hash=content_hash,
    )


@pytest.mark.anyio
async def test_collect_evidence_filters_and_creates_citations() -> None:
    retrieval_service = AsyncMock()
    retrieval_service.retrieve.return_value = RetrievalResult(
        query="heart failure symptoms",
        matches=[
            make_match("chunk-1", 0.91, "same-content"),
            make_match("chunk-2", 0.82, "same-content"),
            make_match("chunk-3", 0.72, "different-content"),
            make_match("chunk-4", 0.20, "weak-content"),
        ],
    )
    service = EvidenceService(retrieval_service)
    document_id = UUID("dd64c66a-19c6-45c0-b665-edd3c3348cd4")

    result = await service.collect_evidence(
        query="heart failure symptoms",
        tenant_id="tenant-1",
        document_id=document_id,
    )

    assert result.query == "heart failure symptoms"
    assert result.sufficient is True
    assert [item.chunk_id for item in result.items] == ["chunk-1", "chunk-3"]
    assert [item.citation_id for item in result.items] == ["C1", "C2"]
    assert result.items[0].text == "Evidence from chunk-1"
    assert result.items[0].title == "Medical Book"
    retrieval_service.retrieve.assert_awaited_once_with(
        query="heart failure symptoms",
        tenant_id="tenant-1",
        top_k=15,
        document_id=str(document_id),
    )


@pytest.mark.anyio
async def test_collect_evidence_uses_chunk_id_when_hash_is_missing() -> None:
    retrieval_service = AsyncMock()
    retrieval_service.retrieve.return_value = RetrievalResult(
        query="query",
        matches=[
            make_match("chunk-1", 0.80),
            make_match("chunk-2", 0.70),
        ],
    )
    service = EvidenceService(retrieval_service)

    result = await service.collect_evidence("query", "tenant-1")

    assert [item.chunk_id for item in result.items] == ["chunk-1", "chunk-2"]


@pytest.mark.anyio
async def test_collect_evidence_respects_evidence_limit() -> None:
    retrieval_service = AsyncMock()
    retrieval_service.retrieve.return_value = RetrievalResult(
        query="query",
        matches=[
            make_match("chunk-1", 0.90),
            make_match("chunk-2", 0.80),
            make_match("chunk-3", 0.70),
        ],
    )
    service = EvidenceService(retrieval_service, evidence_limit=2)

    result = await service.collect_evidence("query", "tenant-1")

    assert [item.chunk_id for item in result.items] == ["chunk-1", "chunk-2"]


@pytest.mark.anyio
async def test_collect_evidence_marks_weak_results_as_insufficient() -> None:
    retrieval_service = AsyncMock()
    retrieval_service.retrieve.return_value = RetrievalResult(
        query="query",
        matches=[make_match("chunk-1", 0.20)],
    )
    service = EvidenceService(retrieval_service, min_score=0.35)

    result = await service.collect_evidence("query", "tenant-1")

    assert result.items == []
    assert result.sufficient is False
