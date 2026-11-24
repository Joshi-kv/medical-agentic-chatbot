from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies.api_keys import get_api_key_service
from api.schemas.api_keys import (
    APIKeyCreatedResponse,
    APIKeyIdentityResponse,
    CreateAPIKeyRequest,
)
from api.security.api_key import AuthenticatedPrincipal
from config.settings import Settings, get_settings
from services.api_keys import APIKeyError, APIKeyService, InvalidAPIKeyInput

router = APIRouter(prefix="/v1/auth", tags=["authentication"])


@router.post(
    "/api-keys",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_api_key(
    request: CreateAPIKeyRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[APIKeyService, Depends(get_api_key_service)],
) -> APIKeyCreatedResponse:
    if not settings.api_key_signup_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "api_key_signup_disabled",
                "message": "API-key signup is disabled.",
            },
        )
    try:
        issued = service.issue(request.name, settings.api_key_tenant_id)
    except InvalidAPIKeyInput as error:
        raise HTTPException(
            status_code=400, detail={"code": "invalid_name", "message": str(error)}
        ) from error
    except APIKeyError as error:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "api_key_service_unavailable",
                "message": "API-key creation is unavailable.",
            },
        ) from error
    return APIKeyCreatedResponse(
        api_key=issued.api_key,
        key_id=issued.record.key_id,
        key_prefix=issued.record.key_prefix,
        name=issued.record.name,
        created_at=issued.record.created_at,
    )


@router.get("/me", response_model=APIKeyIdentityResponse)
def get_identity(principal: AuthenticatedPrincipal) -> APIKeyIdentityResponse:
    return APIKeyIdentityResponse(
        key_id=principal.key_id,
        tenant_id=principal.tenant_id,
        scopes=sorted(principal.scopes),
    )
