from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import SecretStr

from api.dependencies import get_document_service
from api.routers.documents import router
from config.settings import get_settings
from main import app as production_app
from services.documents import (
    DocumentNotFound,
    IdempotencyConflict,
    InvalidDocumentType,
    UploadTooLarge,
)


@dataclass(frozen=True)
class AcceptedDocument:
    document_id: str
    status: str


@dataclass(frozen=True)
class DocumentProgress:
    pages_processed: int
    pages_requiring_ocr: int
    chunks_created: int
    vectors_upserted: int


@dataclass(frozen=True)
class DocumentView:
    document_id: str
    filename: str
    title: str | None
    status: str
    progress: DocumentProgress
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class SuccessfulDocumentService:
    def __init__(self) -> None:
        self.received_content = b""
        self.received_tenant_id = ""
        self.deleted_document_id = ""

    async def submit_document(
        self,
        *,
        content: AsyncIterator[bytes],
        filename: str,
        declared_content_type: str | None,
        title: str | None,
        tenant_id: str,
        idempotency_key: str,
        actor_id: str,
    ) -> AcceptedDocument:
        self.received_content = b"".join([chunk async for chunk in content])
        self.received_tenant_id = tenant_id

        assert filename == "guide.pdf"
        assert declared_content_type == "application/pdf"
        assert title == "Clinical Guide"
        assert idempotency_key == "idempotency-key-123"
        assert actor_id == "test-key"

        return AcceptedDocument(document_id="document-123", status="queued")

    async def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
    ) -> DocumentView:
        assert document_id == "document-123"
        assert tenant_id == "tenant-123"
        timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        return DocumentView(
            document_id=document_id,
            filename="guide.pdf",
            title="Clinical Guide",
            status="processing",
            progress=DocumentProgress(
                pages_processed=4,
                pages_requiring_ocr=1,
                chunks_created=12,
                vectors_upserted=8,
            ),
            error_code=None,
            error_message=None,
            created_at=timestamp,
            updated_at=timestamp,
        )

    async def request_deletion(
        self,
        *,
        document_id: str,
        tenant_id: str,
        actor_id: str,
    ) -> None:
        assert tenant_id == "tenant-123"
        assert actor_id == "test-key"
        self.deleted_document_id = document_id


class FailingUploadService(SuccessfulDocumentService):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def submit_document(self, **kwargs: object) -> AcceptedDocument:
        raise self.error


class MissingDocumentService(SuccessfulDocumentService):
    async def get_document(self, **kwargs: object) -> DocumentView:
        raise DocumentNotFound

    async def request_deletion(self, **kwargs: object) -> None:
        raise DocumentNotFound


def make_client(service: object) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.document_service = service

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


def test_upload_returns_202_and_queues_trusted_tenant() -> None:
    service = SuccessfulDocumentService()
    client = make_client(service)

    response = client.post(
        "/v1/documents",
        headers={
            "X-API-Key": "test-secret",
            "Idempotency-Key": "idempotency-key-123",
        },
        files={"file": ("guide.pdf", b"%PDF-1.7\nexample", "application/pdf")},
        data={"title": "Clinical Guide"},
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json() == {
        "document_id": "document-123",
        "status": "queued",
        "status_url": "/v1/documents/document-123",
    }
    assert response.headers["Location"] == "/v1/documents/document-123"
    assert response.headers["Cache-Control"] == "no-store"
    assert service.received_content == b"%PDF-1.7\nexample"
    assert service.received_tenant_id == "tenant-123"


def test_get_document_returns_tenant_scoped_status() -> None:
    client = make_client(SuccessfulDocumentService())

    response = client.get(
        "/v1/documents/document-123",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["document_id"] == "document-123"
    assert response.json()["status"] == "processing"
    assert response.json()["progress"] == {
        "pages_processed": 4,
        "pages_requiring_ocr": 1,
        "chunks_created": 12,
        "vectors_upserted": 8,
    }


def test_delete_document_returns_202_and_requests_deletion() -> None:
    service = SuccessfulDocumentService()
    client = make_client(service)

    response = client.delete(
        "/v1/documents/document-123",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == status.HTTP_202_ACCEPTED
    assert response.json() == {
        "document_id": "document-123",
        "status": "deleting",
    }
    assert service.deleted_document_id == "document-123"


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (InvalidDocumentType(), 415, "unsupported_media_type"),
        (UploadTooLarge(), 413, "upload_too_large"),
        (IdempotencyConflict(), 409, "idempotency_conflict"),
    ],
)
def test_upload_maps_domain_errors_to_safe_http_responses(
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    client = make_client(FailingUploadService(error))

    response = client.post(
        "/v1/documents",
        headers={
            "X-API-Key": "test-secret",
            "Idempotency-Key": "idempotency-key-123",
        },
        files={"file": ("guide.pdf", b"contents", "application/pdf")},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert "contents" not in response.text


@pytest.mark.parametrize("method", ["get", "delete"])
def test_missing_document_returns_tenant_safe_404(method: str) -> None:
    client = make_client(MissingDocumentService())

    response = getattr(client, method)(
        "/v1/documents/missing-document",
        headers={"X-API-Key": "test-secret"},
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == {
        "code": "document_not_found",
        "message": "Document not found.",
    }


def test_production_app_registers_document_routes() -> None:
    service = SuccessfulDocumentService()
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
    production_app.dependency_overrides[get_document_service] = lambda: service

    try:
        client = TestClient(production_app)
        response = client.get(
            "/v1/documents/document-123",
            headers={"X-API-Key": "test-secret"},
        )
    finally:
        production_app.dependency_overrides.clear()

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["document_id"] == "document-123"
