from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.dependencies.api_keys import get_api_key_service
from api.routers.api_keys import router
from api.security.api_key import APIPrincipal, authenticate_api_key
from config.settings import Settings, get_settings
from services.api_keys import PATIENT_SCOPES, APIKeyRecord, APIKeyService, IssuedAPIKey


def make_client(signup_enabled: bool = True) -> tuple[TestClient, MagicMock]:
    app = FastAPI()
    app.include_router(router)
    settings = MagicMock(spec=Settings)
    settings.api_key_signup_enabled = signup_enabled
    settings.api_key_tenant_id = "tenant-1"
    service = MagicMock(spec=APIKeyService)
    record = APIKeyRecord(
        key_id="key-1",
        tenant_id="tenant-1",
        name="Maya",
        key_prefix="med_example1",
        key_hash="a" * 64,
        scopes=PATIENT_SCOPES,
        created_at=datetime(2026, 11, 23, tzinfo=timezone.utc),
    )
    service.issue.return_value = IssuedAPIKey(api_key="med_example-key", record=record)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_api_key_service] = lambda: service
    app.dependency_overrides[authenticate_api_key] = lambda: APIPrincipal(
        key_id="key-1",
        tenant_id="tenant-1",
        scopes=PATIENT_SCOPES,
    )
    return TestClient(app), service


def test_create_api_key_returns_key_once() -> None:
    client, service = make_client()

    response = client.post("/v1/auth/api-keys", json={"name": "Maya"})

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["api_key"] == "med_example-key"
    service.issue.assert_called_once_with("Maya", "tenant-1")


def test_create_api_key_is_disabled_by_setting() -> None:
    client, service = make_client(signup_enabled=False)

    response = client.post("/v1/auth/api-keys", json={"name": "Maya"})

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"]["code"] == "api_key_signup_disabled"
    service.issue.assert_not_called()


def test_get_identity_returns_authenticated_key() -> None:
    client, _ = make_client()

    response = client.get("/v1/auth/me")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "key_id": "key-1",
        "tenant_id": "tenant-1",
        "scopes": ["conversations:read", "conversations:write"],
    }
