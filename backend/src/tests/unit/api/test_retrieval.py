from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.dependencies.retrieval import get_retrieval_service
from api.routers.retrieval import router
from api.security.api_key import APIPrincipal, authenticate_api_key
from config.settings import get_settings
from main import app as production_app
from services.retrieval import (
    InvalidRetrievalQuery,
    RetrievalMatch,
    RetrievalResult,
    RetrievalService,
    RetrievalServiceError,
)


def make_sample_result(query: str = "cardiac arrest") -> RetrievalResult:
    return RetrievalResult(
        query=query,
        matches=[
            RetrievalMatch(
                chunk_id="chunk-1",
                score=0.95,
                text="Cardiac arrest requires immediate CPR.",
                document_id="doc-123",
                page_number=5,
                chunk_number=1,
                source="handbook.pdf",
                title="Emergency Handbook",
            ),
            RetrievalMatch(
                chunk_id="chunk-2",
                score=0.88,
                text="Defibrillation is indicated for ventricular fibrillation.",
                document_id="doc-123",
                page_number=6,
                chunk_number=0,
                source="handbook.pdf",
                title=None,
            ),
        ],
    )


def make_client(service: object) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_retrieval_service] = lambda: service

    settings = type(
        "TestSettings",
        (),
        {
            "ingestion_api_key": SecretStr("test-secret"),
            "api_key_tenant_id": "tenant-123",
            "api_key_id": "test-key",
        },
    )()
    app.dependency_overrides[get_settings] = lambda: settings

    return TestClient(app, raise_server_exceptions=False)


def test_search_documents_success() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    sample_result = make_sample_result()
    mock_service.retrieve.return_value = sample_result

    client = make_client(mock_service)
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "test-secret"},
        json={"query": "cardiac arrest", "top_k": 5},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["query"] == "cardiac arrest"
    assert len(data["matches"]) == 2
    assert data["matches"][0] == {
        "chunk_id": "chunk-1",
        "score": 0.95,
        "text": "Cardiac arrest requires immediate CPR.",
        "document_id": "doc-123",
        "page_number": 5,
        "chunk_number": 1,
        "source": "handbook.pdf",
        "title": "Emergency Handbook",
    }
    assert data["matches"][1]["title"] is None

    mock_service.retrieve.assert_awaited_once_with(
        query="cardiac arrest",
        tenant_id="tenant-123",
        top_k=5,
        document_id=None,
    )


def test_search_documents_with_document_id() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    mock_service.retrieve.return_value = make_sample_result()

    client = make_client(mock_service)
    target_uuid = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "test-secret"},
        json={
            "query": "cardiac arrest",
            "top_k": 3,
            "document_id": target_uuid,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    mock_service.retrieve.assert_awaited_once_with(
        query="cardiac arrest",
        tenant_id="tenant-123",
        top_k=3,
        document_id=target_uuid,
    )


def test_search_documents_tenant_isolation_uses_principal_tenant() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    mock_service.retrieve.return_value = make_sample_result()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    # Custom tenant in settings
    settings = type(
        "TestSettings",
        (),
        {
            "ingestion_api_key": SecretStr("hospital-a-key"),
            "api_key_tenant_id": "hospital-alpha",
            "api_key_id": "key-alpha",
        },
    )()
    app.dependency_overrides[get_settings] = lambda: settings

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "hospital-a-key"},
        json={"query": "aspirin dosage"},
    )

    assert response.status_code == status.HTTP_200_OK
    mock_service.retrieve.assert_awaited_once_with(
        query="aspirin dosage",
        tenant_id="hospital-alpha",
        top_k=5,
        document_id=None,
    )


def test_search_documents_invalid_query_returns_400() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    mock_service.retrieve.side_effect = InvalidRetrievalQuery("Query cannot be empty.")

    client = make_client(mock_service)
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "test-secret"},
        json={"query": "   "},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json() == {
        "detail": {
            "code": "invalid_retrieval_query",
            "message": "Query cannot be empty.",
        }
    }


def test_search_documents_service_error_returns_503() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    mock_service.retrieve.side_effect = RetrievalServiceError("Index unavailable.")

    client = make_client(mock_service)
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "test-secret"},
        json={"query": "arrhythmia treatments"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json() == {
        "detail": {
            "code": "retrieval_unavailable",
            "message": "Retrieval is temporarily unavailable.",
        }
    }


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"query": ""},  # min_length=1
        {"query": "valid query", "top_k": 0},  # ge=1
        {"query": "valid query", "top_k": 21},  # le=20
        {"query": "valid query", "document_id": "not-a-valid-uuid"},
        {},  # missing query
    ],
)
def test_search_documents_request_validation_returns_422(
    invalid_payload: dict,
) -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    client = make_client(mock_service)

    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "test-secret"},
        json=invalid_payload,
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    mock_service.retrieve.assert_not_called()


def test_search_documents_unauthenticated_returns_401() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    client = make_client(mock_service)

    # Missing API key
    response = client.post(
        "/v1/retrieval/search",
        json={"query": "cardiac arrest"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "invalid_api_key"

    # Wrong API key
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "wrong-secret"},
        json={"query": "cardiac arrest"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "invalid_api_key"


def test_search_documents_insufficient_scope_returns_403() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    # Authenticate as principal with only documents:write, missing documents:read
    restricted_principal = APIPrincipal(
        key_id="test-key",
        tenant_id="tenant-123",
        scopes=frozenset({"documents:write"}),
    )
    app.dependency_overrides[authenticate_api_key] = lambda: restricted_principal

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/retrieval/search",
        headers={"X-API-Key": "test-secret"},
        json={"query": "cardiac arrest"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == {
        "code": "insufficient_scope",
        "message": "The credential cannot perform this action.",
    }


def test_production_app_registers_retrieval_routes() -> None:
    mock_service = AsyncMock(spec=RetrievalService)
    mock_service.retrieve.return_value = make_sample_result()

    settings = type(
        "TestSettings",
        (),
        {
            "ingestion_api_key": SecretStr("test-secret"),
            "api_key_tenant_id": "tenant-123",
            "api_key_id": "test-key",
        },
    )()
    production_app.dependency_overrides[get_settings] = lambda: settings
    production_app.dependency_overrides[get_retrieval_service] = lambda: mock_service

    try:
        client = TestClient(production_app)
        response = client.post(
            "/v1/retrieval/search",
            headers={"X-API-Key": "test-secret"},
            json={"query": "cardiac arrest"},
        )
    finally:
        production_app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["query"] == "cardiac arrest"
