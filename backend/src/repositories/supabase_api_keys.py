from datetime import datetime
from typing import Any

from supabase import Client

from services.api_keys import APIKeyRecord


class SupabaseAPIKeyStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    def create(self, record: APIKeyRecord) -> APIKeyRecord:
        response = (
            self._client.table("api_keys")
            .insert(
                {
                    "key_id": record.key_id,
                    "tenant_id": record.tenant_id,
                    "name": record.name,
                    "key_prefix": record.key_prefix,
                    "key_hash": record.key_hash,
                    "scopes": sorted(record.scopes),
                    "created_at": record.created_at.isoformat(),
                }
            )
            .execute()
        )
        return _from_row(_first_row(response.data))

    def find_active(self, key_hash: str) -> APIKeyRecord | None:
        response = (
            self._client.table("api_keys")
            .select("key_id,tenant_id,name,key_prefix,key_hash,scopes,created_at")
            .eq("key_hash", key_hash)
            .is_("revoked_at", "null")
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return _from_row(_first_row(response.data))


def _from_row(row: dict[str, Any]) -> APIKeyRecord:
    scopes = row.get("scopes")
    created_at = row.get("created_at")
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) for scope in scopes
    ):
        raise RuntimeError("Supabase returned invalid API-key scopes.")
    if not isinstance(created_at, str):
        raise RuntimeError("Supabase returned an invalid API-key timestamp.")
    return APIKeyRecord(
        key_id=str(row["key_id"]),
        tenant_id=str(row["tenant_id"]),
        name=str(row["name"]),
        key_prefix=str(row["key_prefix"]),
        key_hash=str(row["key_hash"]),
        scopes=frozenset(scopes),
        created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
    )


def _first_row(data: object) -> dict[str, Any]:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    raise RuntimeError("Supabase returned an unexpected API-key response.")
