from datetime import datetime
from typing import Any

from anyio import to_thread

from services.documents import (
    CreateDocumentCommand,
    CreateDocumentResult,
    DocumentProgress,
    DocumentRecord,
    IdempotencyConflict,
)


class SupabaseDocumentStore:
    def __init__(self, *, client: Any) -> None:
        self._client = client

    async def create_queued_document(
        self,
        command: CreateDocumentCommand,
    ) -> CreateDocumentResult:
        params: dict[str, object] = {
            "p_document_id": command.document_id,
            "p_tenant_id": command.tenant_id,
            "p_filename": command.filename,
            "p_title": command.title,
            "p_content_type": command.content_type,
            "p_size_bytes": command.size_bytes,
            "p_checksum": command.checksum,
            "p_storage_key": command.storage_key,
            "p_idempotency_key": command.idempotency_key,
            "p_embedding_model": command.embedding_model,
            "p_embedding_dimension": command.embedding_dimension,
            "p_pipeline_version": command.pipeline_version,
            "p_actor_id": command.actor_id,
            "p_created_at": command.created_at.isoformat(),
        }
        try:
            response = await to_thread.run_sync(
                lambda: self._client.rpc(
                    "create_document_ingestion",
                    params,
                ).execute()
            )
        except Exception as error:
            if "idempotency_conflict" in str(error).lower():
                raise IdempotencyConflict from error
            raise

        row = _first_row(response.data)
        return CreateDocumentResult(
            document_id=str(row["document_id"]),
            status=str(row["status"]),
            created=bool(row["created"]),
        )

    async def get_document(
        self,
        *,
        document_id: str,
        tenant_id: str,
    ) -> DocumentRecord | None:
        response = await to_thread.run_sync(
            lambda: self._client.table("documents")
            .select(
                "document_id,filename,title,status,pages_processed,"
                "pages_requiring_ocr,chunks_created,vectors_upserted,"
                "error_code,error_message,created_at,updated_at"
            )
            .eq("document_id", document_id)
            .eq("tenant_id", tenant_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        row = _first_row(response.data)
        return DocumentRecord(
            document_id=str(row["document_id"]),
            filename=str(row["filename"]),
            title=_optional_string(row.get("title")),
            status=str(row["status"]),
            progress=DocumentProgress(
                pages_processed=int(row["pages_processed"]),
                pages_requiring_ocr=int(row["pages_requiring_ocr"]),
                chunks_created=int(row["chunks_created"]),
                vectors_upserted=int(row["vectors_upserted"]),
            ),
            error_code=_optional_string(row.get("error_code")),
            error_message=_optional_string(row.get("error_message")),
            created_at=_parse_datetime(row["created_at"]),
            updated_at=_parse_datetime(row["updated_at"]),
        )

    async def request_deletion(
        self,
        *,
        document_id: str,
        tenant_id: str,
        actor_id: str,
    ) -> bool:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "request_document_deletion",
                {
                    "p_document_id": document_id,
                    "p_tenant_id": tenant_id,
                    "p_actor_id": actor_id,
                },
            ).execute()
        )
        data = response.data
        if isinstance(data, list):
            return bool(data[0]) if data else False
        return bool(data)


def _first_row(data: object) -> dict[str, Any]:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    raise RuntimeError("Supabase returned an unexpected document response")


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError("Supabase returned an invalid timestamp")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
