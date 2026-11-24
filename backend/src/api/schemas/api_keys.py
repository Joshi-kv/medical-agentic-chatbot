from datetime import datetime

from pydantic import BaseModel, Field


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)


class APIKeyCreatedResponse(BaseModel):
    api_key: str
    key_id: str
    key_prefix: str
    name: str
    created_at: datetime


class APIKeyIdentityResponse(BaseModel):
    key_id: str
    tenant_id: str
    scopes: list[str]
