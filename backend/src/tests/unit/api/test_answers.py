from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.dependencies.answering import get_answering_service
from api.routers.answering import router
from api.security.api_key import APIPrincipal, authenticate_api_key
from config.settings import get_settings
from main import app as production_app
from services.answering import (
    INSUFFICIENT_MESSAGE,
    SAFETY_NOTICE,
    AnsweringError,
    AnsweringService,
    AnswerResult,
)
from services.evidence import Evidence
from services.retrieval import RetrievalServiceError


def make_sample_evidence(
    citation_id: str = "C1",
    chunk_id: str = "chunk-1",
    text: str = "Hypertension increases risk of stroke.",
    document_id: str = "doc-123",
    page_number: int = 5,
    source: str = "handbook.pdf",
    title: str | None = "Emergency Handbook",
) -> Evidence:
    return Evidence(
        citation_id=citation_id,
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        page_number=page_number,
        source=source,
        score=0.95,
        title=title,
    )


def make_sample_result(
    answer: str = "Hypertension can cause stroke [C1].",
    citations: list[Evidence] | None = None,
    sufficient_evidence: bool = True,
) -> AnswerResult:
    if citations is None:
        citations = [make_sample_evidence()]
    return AnswerResult(
        answer=answer,
        citations=citations,
        sufficient_evidence=sufficient_evidence,
        safety_notice=SAFETY_NOTICE,
    )


def make_client(service: object) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_answering_service] = lambda: service

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


def test_create_answer_success() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    sample_result = make_sample_result(
        answer="Hypertension can lead to stroke [C1] and heart failure [C2].",
        citations=[
            make_sample_evidence(citation_id="C1", title="Clinical Cardiology"),
            make_sample_evidence(
                citation_id="C2",
                chunk_id="chunk-2",
                title=None,
                source="guidelines.pdf",
                page_number=12,
            ),
        ],
    )
    mock_service.answer.return_value = sample_result

    client = make_client(mock_service)
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "test-secret"},
        json={"query": "What are complications of hypertension?"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == sample_result.answer
    assert data["sufficient_evidence"] is True
    assert data["safety_notice"] == SAFETY_NOTICE
    assert len(data["citations"]) == 2
    assert data["citations"][0] == {
        "citation_id": "C1",
        "document_id": "doc-123",
        "chunk_id": "chunk-1",
        "page_number": 5,
        "source": "handbook.pdf",
        "title": "Clinical Cardiology",
    }
    assert data["citations"][1]["title"] is None
    assert data["citations"][1]["source"] == "guidelines.pdf"

    mock_service.answer.assert_awaited_once_with(
        query="What are complications of hypertension?",
        tenant_id="tenant-123",
        document_id=None,
    )


def test_create_answer_with_document_id() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    mock_service.answer.return_value = make_sample_result()

    client = make_client(mock_service)
    target_uuid = "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "test-secret"},
        json={
            "query": "What are the side effects?",
            "document_id": target_uuid,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    mock_service.answer.assert_awaited_once_with(
        query="What are the side effects?",
        tenant_id="tenant-123",
        document_id=UUID(target_uuid),
    )


def test_create_answer_tenant_isolation_uses_principal_tenant() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    mock_service.answer.return_value = make_sample_result()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_answering_service] = lambda: mock_service

    settings = type(
        "TestSettings",
        (),
        {
            "ingestion_api_key": SecretStr("hospital-alpha-key"),
            "api_key_tenant_id": "hospital-alpha",
            "api_key_id": "key-alpha",
        },
    )()
    app.dependency_overrides[get_settings] = lambda: settings

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "hospital-alpha-key"},
        json={"query": "Aspirin dosage"},
    )

    assert response.status_code == status.HTTP_200_OK
    mock_service.answer.assert_awaited_once_with(
        query="Aspirin dosage",
        tenant_id="hospital-alpha",
        document_id=None,
    )


def test_create_answer_insufficient_evidence_response() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    mock_service.answer.return_value = AnswerResult(
        answer=INSUFFICIENT_MESSAGE,
        citations=[],
        sufficient_evidence=False,
        safety_notice=SAFETY_NOTICE,
    )

    client = make_client(mock_service)
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "test-secret"},
        json={"query": "Obscure medical topic"},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["answer"] == INSUFFICIENT_MESSAGE
    assert data["citations"] == []
    assert data["sufficient_evidence"] is False
    assert data["safety_notice"] == SAFETY_NOTICE


@pytest.mark.parametrize(
    "error",
    [
        AnsweringError("Model generation failed"),
        RetrievalServiceError("Vector search failed"),
    ],
)
def test_create_answer_service_errors_return_503(error: Exception) -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    mock_service.answer.side_effect = error

    client = make_client(mock_service)
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "test-secret"},
        json={"query": "What causes migraines?"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json() == {
        "detail": {
            "code": "answering_unavailable",
            "message": "Answering is temporarily unavailable.",
        }
    }


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"query": ""},  # min_length=1
        {"query": "a" * 2001},  # max_length=2000
        {"query": "valid query", "document_id": "not-a-valid-uuid"},
        {},  # missing query
    ],
)
def test_create_answer_request_validation_returns_422(
    invalid_payload: dict,
) -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    client = make_client(mock_service)

    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "test-secret"},
        json=invalid_payload,
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    mock_service.answer.assert_not_called()


def test_create_answer_unauthenticated_returns_401() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    client = make_client(mock_service)

    # Missing API key
    response = client.post(
        "/v1/answers",
        json={"query": "What is hypertension?"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "invalid_api_key"

    # Wrong API key
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "wrong-secret"},
        json={"query": "What is hypertension?"},
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "invalid_api_key"


def test_create_answer_insufficient_scope_returns_403() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_answering_service] = lambda: mock_service

    # Authenticate as principal missing documents:read
    restricted_principal = APIPrincipal(
        key_id="test-key",
        tenant_id="tenant-123",
        scopes=frozenset({"documents:write"}),
    )
    app.dependency_overrides[authenticate_api_key] = lambda: restricted_principal

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/v1/answers",
        headers={"X-API-Key": "test-secret"},
        json={"query": "What is hypertension?"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == {
        "code": "insufficient_scope",
        "message": "The credential cannot perform this action.",
    }


def test_production_app_registers_answering_routes() -> None:
    mock_service = AsyncMock(spec=AnsweringService)
    mock_service.answer.return_value = make_sample_result()

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
    production_app.dependency_overrides[get_answering_service] = (
        lambda: mock_service
    )

    try:
        client = TestClient(production_app)
        response = client.post(
            "/v1/answers",
            headers={"X-API-Key": "test-secret"},
            json={"query": "What is hypertension?"},
        )
    finally:
        production_app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["answer"] == "Hypertension can cause stroke [C1]."
