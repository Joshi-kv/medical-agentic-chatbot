from datetime import datetime, timezone

import pytest

from repositories.supabase_documents import SupabaseDocumentStore
from services.documents import CreateDocumentCommand
from storage.supabase_documents import SupabaseDocumentStorage


class Response:
    def __init__(self, data: object) -> None:
        self.data = data


class Query:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.filters: list[tuple[str, object]] = []

    def select(self, _columns: str) -> "Query":
        return self

    def eq(self, column: str, value: object) -> "Query":
        self.filters.append((column, value))
        return self

    def limit(self, _count: int) -> "Query":
        return self

    def execute(self) -> Response:
        return self.response


class FakeBucket:
    def __init__(self) -> None:
        self.upload_call: tuple[str, bytes, dict[str, str]] | None = None
        self.removed: list[str] | None = None
        self.downloaded_path: str | None = None

    def upload(
        self,
        path: str,
        file: bytes,
        file_options: dict[str, str],
    ) -> None:
        self.upload_call = (path, file, file_options)

    def remove(self, paths: list[str]) -> None:
        self.removed = paths

    def download(self, path: str) -> bytes:
        self.downloaded_path = path
        return b"%PDF-1.7\nbody"


class FakeStorageRoot:
    def __init__(self, bucket: FakeBucket) -> None:
        self.bucket = bucket
        self.bucket_name: str | None = None

    def from_(self, bucket_name: str) -> FakeBucket:
        self.bucket_name = bucket_name
        return self.bucket


class FakeClient:
    def __init__(self) -> None:
        self.bucket = FakeBucket()
        self.storage = FakeStorageRoot(self.bucket)
        self.rpc_calls: list[tuple[str, dict[str, object]]] = []
        self.rpc_responses: dict[str, Response] = {}
        self.table_query = Query(Response([]))

    def rpc(self, name: str, params: dict[str, object]) -> Query:
        self.rpc_calls.append((name, params))
        return Query(self.rpc_responses[name])

    def table(self, _name: str) -> Query:
        return self.table_query


@pytest.mark.anyio
async def test_supabase_storage_uploads_private_pdf_without_upsert() -> None:
    client = FakeClient()
    storage = SupabaseDocumentStorage(client=client, bucket="medical-documents")

    await storage.upload(
        storage_key="tenants/t/documents/d/original.pdf",
        content=b"pdf",
    )
    await storage.delete(storage_key="tenants/t/documents/d/original.pdf")

    assert client.storage.bucket_name == "medical-documents"
    assert client.bucket.upload_call == (
        "tenants/t/documents/d/original.pdf",
        b"pdf",
        {"content-type": "application/pdf", "cache-control": "3600", "upsert": "false"},
    )
    assert client.bucket.removed == ["tenants/t/documents/d/original.pdf"]


@pytest.mark.anyio
async def test_supabase_storage_downloads_private_pdf() -> None:
    client = FakeClient()
    storage = SupabaseDocumentStorage(client=client, bucket="medical-documents")

    content = await storage.download(
        storage_key="tenants/t/documents/d/original.pdf"
    )

    assert content == b"%PDF-1.7\nbody"
    assert client.bucket.downloaded_path == "tenants/t/documents/d/original.pdf"


@pytest.mark.anyio
async def test_supabase_store_calls_atomic_creation_rpc() -> None:
    client = FakeClient()
    client.rpc_responses["create_document_ingestion"] = Response(
        [{"document_id": "document-1", "status": "queued", "created": True}]
    )
    store = SupabaseDocumentStore(client=client)
    command = CreateDocumentCommand(
        document_id="document-1",
        tenant_id="tenant-1",
        filename="guide.pdf",
        title="Guide",
        content_type="application/pdf",
        size_bytes=3,
        checksum="a" * 64,
        storage_key="tenants/tenant-1/documents/document-1/original.pdf",
        idempotency_key="idempotency-key-1",
        embedding_model="embedding-model",
        embedding_dimension=768,
        pipeline_version="1",
        actor_id="primary",
        created_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
    )

    result = await store.create_queued_document(command)

    assert result.created is True
    assert result.document_id == "document-1"
    name, params = client.rpc_calls[0]
    assert name == "create_document_ingestion"
    assert params["p_tenant_id"] == "tenant-1"
    assert params["p_checksum"] == "a" * 64


@pytest.mark.anyio
async def test_supabase_store_maps_document_status_record() -> None:
    client = FakeClient()
    client.table_query = Query(
        Response(
            [
                {
                    "document_id": "document-1",
                    "filename": "guide.pdf",
                    "title": None,
                    "status": "processing",
                    "pages_processed": 2,
                    "pages_requiring_ocr": 1,
                    "chunks_created": 8,
                    "vectors_upserted": 8,
                    "error_code": None,
                    "error_message": None,
                    "created_at": "2026-11-03T10:00:00+00:00",
                    "updated_at": "2026-11-03T10:01:00+00:00",
                }
            ]
        )
    )
    store = SupabaseDocumentStore(client=client)

    record = await store.get_document(document_id="document-1", tenant_id="tenant-1")

    assert record is not None
    assert record.progress.chunks_created == 8
    assert ("document_id", "document-1") in client.table_query.filters
    assert ("tenant_id", "tenant-1") in client.table_query.filters
