from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict

from config.settings import Settings, get_settings

api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description="API key required to access protected endpoints.",
    auto_error=False,
)

staff_api_key_header = APIKeyHeader(
    name="X-Staff-API-Key",
    scheme_name="StaffApiKeyAuth",
    description="Staff API key required for human-review operations.",
    auto_error=False,
)


class APIPrincipal(BaseModel):
    """Server trusted identity with an API Key."""

    model_config = ConfigDict(frozen=True)

    key_id: str
    tenant_id: str
    scopes: frozenset[str]


def authenticate_api_key(
    provided_key: Annotated[str | None, Security(api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Authenticates an API key provided in the X-API-Key header."""

    expected_key = settings.ingestion_api_key.get_secret_value()

    if (
        provided_key is None
        or not expected_key
        or not compare_digest(provided_key, expected_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid or missing API key.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return APIPrincipal(
        key_id=settings.api_key_id,
        tenant_id=settings.api_key_tenant_id,
        scopes=frozenset(
            {
                "documents:read",
                "documents:write",
                "documents:delete",
                "conversations:read",
                "conversations:write",
            }
        ),
    )


def authenticate_staff_api_key(
    provided_key: Annotated[str | None, Security(staff_api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> APIPrincipal:
    configured_key = settings.staff_api_key

    if configured_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "staff_auth_unavailable",
                "message": "Staff authentication is not configured.",
            },
        )

    expected_key = configured_key.get_secret_value()

    if (
        provided_key is None
        or not expected_key
        or not compare_digest(provided_key, expected_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "invalid_staff_api_key",
                "message": "Invalid or missing staff API key.",
            },
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return APIPrincipal(
        key_id=settings.staff_api_key_id,
        tenant_id=settings.api_key_tenant_id,
        scopes=frozenset(
            {
                "human_review:read",
                "human_review:write",
            }
        ),
    )


AuthenticatedPrincipal = Annotated[
    APIPrincipal,
    Depends(authenticate_api_key),
]


StaffPrincipal = Annotated[
    APIPrincipal,
    Depends(authenticate_staff_api_key),
]
