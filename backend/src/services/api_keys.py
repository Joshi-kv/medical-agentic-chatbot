from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from secrets import token_urlsafe
from typing import Protocol
from uuid import uuid4

PATIENT_SCOPES = frozenset({"conversations:read", "conversations:write"})


class APIKeyError(Exception):
    pass


class InvalidAPIKeyInput(APIKeyError):
    pass


@dataclass(frozen=True)
class APIKeyRecord:
    key_id: str
    tenant_id: str
    name: str
    key_prefix: str
    key_hash: str
    scopes: frozenset[str]
    created_at: datetime


@dataclass(frozen=True)
class IssuedAPIKey:
    api_key: str
    record: APIKeyRecord


class APIKeyStore(Protocol):
    def create(self, record: APIKeyRecord) -> APIKeyRecord: ...

    def find_active(self, key_hash: str) -> APIKeyRecord | None: ...


class APIKeyService:
    def __init__(self, store: APIKeyStore) -> None:
        self._store = store

    def issue(self, name: str, tenant_id: str) -> IssuedAPIKey:
        normalized_name = name.strip()
        if len(normalized_name) < 2 or len(normalized_name) > 100:
            raise InvalidAPIKeyInput("name must be between 2 and 100 characters.")
        raw_key = f"med_{token_urlsafe(32)}"
        record = APIKeyRecord(
            key_id=str(uuid4()),
            tenant_id=tenant_id,
            name=normalized_name,
            key_prefix=raw_key[:12],
            key_hash=self.hash_key(raw_key),
            scopes=PATIENT_SCOPES,
            created_at=datetime.now(timezone.utc),
        )
        try:
            created = self._store.create(record)
        except Exception as error:
            raise APIKeyError("Unable to create API key.") from error
        return IssuedAPIKey(api_key=raw_key, record=created)

    def authenticate(self, raw_key: str) -> APIKeyRecord | None:
        try:
            return self._store.find_active(self.hash_key(raw_key))
        except Exception as error:
            raise APIKeyError("Unable to validate API key.") from error

    @staticmethod
    def hash_key(raw_key: str) -> str:
        return sha256(raw_key.encode("utf-8")).hexdigest()
