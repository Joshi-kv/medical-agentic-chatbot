from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.retrieval import (
    InvalidRetrievalQuery,
    RetrievalService,
    RetrievalServiceError,
)


def make_service() -> tuple[RetrievalService, MagicMock, MagicMock]:
    embeddings = MagicMock()
    index = MagicMock()
    service = RetrievalService(
        embeddings=embeddings,
        index=index,
        namespace="development",
    )
    return service, embeddings, index


@pytest.mark.anyio
async def test_retrieve_returns_matching_chunks() -> None:
    service, embeddings, index = make_service()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                id="chunk-1",
                score=0.91,
                metadata={
                    "text": "Heart failure may cause shortness of breath.",
                    "document_id": "document-1",
                    "page_number": 12,
                    "chunk_number": 2,
                    "source": "medical-book.pdf",
                    "title": "Medical Book",
                },
            )
        ]
    )

    result = await service.retrieve(
        "  symptoms of heart failure  ",
        tenant_id="tenant-1",
        top_k=5,
        document_id="document-1",
    )

    assert result.query == "symptoms of heart failure"
    assert len(result.matches) == 1
    assert result.matches[0].chunk_id == "chunk-1"
    assert result.matches[0].page_number == 12
    embeddings.embed_query.assert_called_once_with("symptoms of heart failure")
    index.query.assert_called_once_with(
        vector=[0.1, 0.2, 0.3],
        top_k=5,
        include_metadata=True,
        namespace="development",
        filter={
            "tenant_id": {"$eq": "tenant-1"},
            "document_id": {"$eq": "document-1"},
        },
    )


@pytest.mark.anyio
async def test_retrieve_always_filters_by_tenant() -> None:
    service, embeddings, index = make_service()
    embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    index.query.return_value = SimpleNamespace(matches=[])

    await service.retrieve("heart health", tenant_id="tenant-1")

    assert index.query.call_args.kwargs["filter"] == {
        "tenant_id": {"$eq": "tenant-1"}
    }


@pytest.mark.anyio
@pytest.mark.parametrize("query", ["", "   "])
async def test_retrieve_rejects_empty_queries(query: str) -> None:
    service, embeddings, index = make_service()

    with pytest.raises(InvalidRetrievalQuery, match="cannot be empty"):
        await service.retrieve(query, tenant_id="tenant-1")

    embeddings.embed_query.assert_not_called()
    index.query.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("top_k", [0, 21])
async def test_retrieve_rejects_invalid_top_k(top_k: int) -> None:
    service, _, _ = make_service()

    with pytest.raises(InvalidRetrievalQuery, match="between 1 and 20"):
        await service.retrieve("heart health", "tenant-1", top_k=top_k)


@pytest.mark.anyio
async def test_retrieve_wraps_provider_errors() -> None:
    service, embeddings, _ = make_service()
    embeddings.embed_query.side_effect = RuntimeError("provider unavailable")

    with pytest.raises(RetrievalServiceError, match="Retrieval failed"):
        await service.retrieve("heart health", "tenant-1")
