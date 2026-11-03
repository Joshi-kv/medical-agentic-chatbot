from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from api.dependencies import require_scope
from api.security.api_key import (
    APIPrincipal,
    AuthenticatedPrincipal,
    authenticate_api_key,
)
from config.settings import Settings, get_settings


def make_mock_settings(
    api_key: str = "secret-test-key-123",
    tenant_id: str = "tenant-test",
    key_id: str = "key-test",
) -> Settings:
    mock_settings = MagicMock(spec=Settings)
    mock_settings.ingestion_api_key = SecretStr(api_key)
    mock_settings.api_key_tenant_id = tenant_id
    mock_settings.api_key_id = key_id
    return mock_settings


def test_api_principal_is_frozen() -> None:
    principal = APIPrincipal(
        key_id="k-1",
        tenant_id="t-1",
        scopes=frozenset({"documents:read"}),
    )
    assert principal.key_id == "k-1"
    assert principal.tenant_id == "t-1"
    assert "documents:read" in principal.scopes

    with pytest.raises(ValidationError):
        principal.key_id = "k-2"  # type: ignore[misc]


def test_authenticate_api_key_success() -> None:
    settings = make_mock_settings(
        api_key="valid-secret-key",
        tenant_id="hospital-xyz",
        key_id="primary-key",
    )

    principal = authenticate_api_key(
        provided_key="valid-secret-key",
        settings=settings,
    )

    assert isinstance(principal, APIPrincipal)
    assert principal.key_id == "primary-key"
    assert principal.tenant_id == "hospital-xyz"
    assert principal.scopes == frozenset(
        {"documents:read", "documents:write", "documents:delete"}
    )


def test_authenticate_api_key_missing_key_raises_401() -> None:
    settings = make_mock_settings(api_key="valid-secret-key")

    with pytest.raises(HTTPException) as exc_info:
        authenticate_api_key(provided_key=None, settings=settings)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == {
        "code": "invalid_api_key",
        "message": "Invalid or missing API key.",
    }
    assert exc_info.value.headers == {"WWW-Authenticate": "ApiKey"}


def test_authenticate_api_key_invalid_key_raises_401() -> None:
    settings = make_mock_settings(api_key="valid-secret-key")

    with pytest.raises(HTTPException) as exc_info:
        authenticate_api_key(provided_key="wrong-key", settings=settings)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail["code"] == "invalid_api_key"
    assert exc_info.value.headers == {"WWW-Authenticate": "ApiKey"}


def test_authenticate_api_key_empty_expected_key_raises_401() -> None:
    settings = make_mock_settings(api_key="")

    with pytest.raises(HTTPException) as exc_info:
        authenticate_api_key(provided_key="", settings=settings)

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


def test_require_scope_authorizes_valid_scope() -> None:
    principal = APIPrincipal(
        key_id="k-1",
        tenant_id="t-1",
        scopes=frozenset({"documents:read", "documents:write"}),
    )

    authorizer = require_scope("documents:read")
    result = authorizer(principal)

    assert result == principal


def test_require_scope_rejects_missing_scope() -> None:
    principal = APIPrincipal(
        key_id="k-1",
        tenant_id="t-1",
        scopes=frozenset({"documents:read"}),
    )

    authorizer = require_scope("documents:write")
    with pytest.raises(HTTPException) as exc_info:
        authorizer(principal)

    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert exc_info.value.detail == {
        "code": "insufficient_scope",
        "message": "The credential cannot perform this action.",
    }


# Integration tests via FastAPI TestClient
@pytest.fixture
def app_with_auth() -> FastAPI:
    app = FastAPI()
    test_settings = make_mock_settings(api_key="test-secret-key")
    app.dependency_overrides[get_settings] = lambda: test_settings

    @app.get("/protected")
    def protected_route(principal: AuthenticatedPrincipal) -> dict:
        return {"tenant_id": principal.tenant_id, "key_id": principal.key_id}

    @app.get("/scoped-read")
    def scoped_read(
        principal: APIPrincipal = Depends(require_scope("documents:read")),
    ) -> dict:
        return {"status": "read-authorized"}

    @app.get("/scoped-admin")
    def scoped_admin(
        principal: APIPrincipal = Depends(require_scope("admin:superuser")),
    ) -> dict:
        return {"status": "admin-authorized"}

    return app


def test_http_endpoint_without_api_key_returns_401(app_with_auth: FastAPI) -> None:
    client = TestClient(app_with_auth)
    response = client.get("/protected")

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "invalid_api_key"
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_http_endpoint_with_invalid_api_key_returns_401(
    app_with_auth: FastAPI,
) -> None:
    client = TestClient(app_with_auth)
    response = client.get("/protected", headers={"X-API-Key": "invalid-token"})

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "invalid_api_key"


def test_http_endpoint_with_valid_api_key_returns_200(
    app_with_auth: FastAPI,
) -> None:
    client = TestClient(app_with_auth)
    response = client.get("/protected", headers={"X-API-Key": "test-secret-key"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"tenant_id": "tenant-test", "key_id": "key-test"}


def test_http_scoped_endpoint_with_sufficient_scope(
    app_with_auth: FastAPI,
) -> None:
    client = TestClient(app_with_auth)
    response = client.get("/scoped-read", headers={"X-API-Key": "test-secret-key"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "read-authorized"}


def test_http_scoped_endpoint_with_insufficient_scope(
    app_with_auth: FastAPI,
) -> None:
    client = TestClient(app_with_auth)
    response = client.get("/scoped-admin", headers={"X-API-Key": "test-secret-key"})

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == {
        "code": "insufficient_scope",
        "message": "The credential cannot perform this action.",
    }
