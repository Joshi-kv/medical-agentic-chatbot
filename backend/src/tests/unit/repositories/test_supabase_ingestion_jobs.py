from typing import Any

import pytest

from repositories.supabase_documents import SupabaseDocumentStore
from services.documents import DocumentProgress


class Response:
    def __init__(self, data: object) -> None:
        self.data = data


class Query:
    def __init__(self, response: Response) -> None:
        self._response = response

    def execute(self) -> Response:
        return self._response


class FakeClient:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def rpc(self, name: str, params: dict[str, object]) -> Query:
        self.calls.append((name, params))
        return Query(Response(self._responses[name]))


@pytest.mark.anyio
async def test_claim_returns_none_when_no_job_is_available() -> None:
    store = SupabaseDocumentStore(client=FakeClient({"claim_ingestion_job": []}))

    job = await store.claim_ingestion_job(
        worker_id="worker-1",
        lease_seconds=1800,
    )

    assert job is None


@pytest.mark.anyio
async def test_claim_maps_complete_job_payload() -> None:
    client = FakeClient(
        {
            "claim_ingestion_job": [
                {
                    "job_id": "job-1",
                    "document_id": "document-1",
                    "tenant_id": "tenant-1",
                    "filename": "medical.pdf",
                    "title": "Medical book",
                    "content_type": "application/pdf",
                    "size_bytes": 1234,
                    "checksum": "a" * 64,
                    "storage_key": "tenants/tenant-1/documents/document-1/original.pdf",
                    "attempts": 1,
                    "max_attempts": 5,
                }
            ]
        }
    )
    store = SupabaseDocumentStore(client=client)

    job = await store.claim_ingestion_job(
        worker_id="worker-1",
        lease_seconds=1800,
    )

    assert job is not None
    assert job.document_id == "document-1"
    assert job.title == "Medical book"
    assert job.attempts == 1
    assert client.calls == [
        (
            "claim_ingestion_job",
            {"p_worker_id": "worker-1", "p_lease_seconds": 1800},
        )
    ]


@pytest.mark.anyio
async def test_complete_sends_progress_to_atomic_rpc() -> None:
    client = FakeClient({"complete_ingestion_job": True})
    store = SupabaseDocumentStore(client=client)

    completed = await store.complete_ingestion_job(
        job_id="job-1",
        document_id="document-1",
        tenant_id="tenant-1",
        worker_id="worker-1",
        progress=DocumentProgress(
            pages_processed=20,
            pages_requiring_ocr=2,
            chunks_created=80,
            vectors_upserted=80,
        ),
    )

    assert completed is True
    assert client.calls == [
        (
            "complete_ingestion_job",
            {
                "p_job_id": "job-1",
                "p_document_id": "document-1",
                "p_tenant_id": "tenant-1",
                "p_worker_id": "worker-1",
                "p_pages_processed": 20,
                "p_pages_requiring_ocr": 2,
                "p_chunks_created": 80,
                "p_vectors_upserted": 80,
            },
        )
    ]


@pytest.mark.anyio
async def test_renew_lease_requires_current_job_and_worker() -> None:
    client = FakeClient({"renew_ingestion_job_lease": True})
    store = SupabaseDocumentStore(client=client)

    renewed = await store.renew_ingestion_job_lease(
        job_id="job-1",
        worker_id="worker-1",
    )

    assert renewed is True
    assert client.calls == [
        (
            "renew_ingestion_job_lease",
            {"p_job_id": "job-1", "p_worker_id": "worker-1"},
        )
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("rpc_result", "expected_status"),
    [("queued", "queued"), ("failed", "failed")],
)
async def test_fail_returns_retry_or_terminal_status(
    rpc_result: str,
    expected_status: str,
) -> None:
    client = FakeClient({"fail_ingestion_job": rpc_result})
    store = SupabaseDocumentStore(client=client)

    next_status = await store.fail_ingestion_job(
        job_id="job-1",
        document_id="document-1",
        tenant_id="tenant-1",
        worker_id="worker-1",
        error_code="InvalidPDFError",
        error_message="PDF could not be parsed",
        progress=DocumentProgress(pages_processed=1),
        retry_base_seconds=30,
    )

    assert next_status == expected_status
    name, params = client.calls[0]
    assert name == "fail_ingestion_job"
    expected: dict[str, Any] = {
        "p_job_id": "job-1",
        "p_document_id": "document-1",
        "p_tenant_id": "tenant-1",
        "p_worker_id": "worker-1",
        "p_error_code": "InvalidPDFError",
        "p_error_message": "PDF could not be parsed",
        "p_pages_processed": 1,
        "p_pages_requiring_ocr": 0,
        "p_chunks_created": 0,
        "p_vectors_upserted": 0,
        "p_retry_base_seconds": 30,
    }
    assert params == expected
