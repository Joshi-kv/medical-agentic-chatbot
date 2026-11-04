from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event

import pytest

from ingestion.schemas import IngestionResult, IngestionStatus, SourceDocument
from repositories.supabase_documents import IngestionJob
from services.documents import DocumentProgress
from worker_runtime import (
    DocumentChecksumMismatch,
    DocumentSizeMismatch,
    IngestionWorker,
    InvalidPDFSignature,
    validate_document_content,
)


def make_job() -> IngestionJob:
    return IngestionJob(
        job_id="job-1",
        document_id="document-1",
        tenant_id="tenant-1",
        filename="medical.pdf",
        title="Medical book",
        content_type="application/pdf",
        size_bytes=13,
        checksum="3f972854841afd236b04b5d7435b73216bc5fa6e39a86aff6e492b744086189c",
        storage_key="tenants/tenant-1/documents/document-1/original.pdf",
        attempts=1,
        max_attempts=5,
    )


def test_validate_document_content_accepts_matching_pdf() -> None:
    validate_document_content(make_job(), b"%PDF-1.7\nbody")


def test_validate_document_content_rejects_size_mismatch() -> None:
    with pytest.raises(DocumentSizeMismatch):
        validate_document_content(make_job(), b"%PDF-")


def test_validate_document_content_rejects_checksum_mismatch() -> None:
    job = replace(make_job(), checksum="a" * 64)

    with pytest.raises(DocumentChecksumMismatch):
        validate_document_content(job, b"%PDF-1.7\nbody")


def test_validate_document_content_rejects_invalid_pdf_signature() -> None:
    job = replace(
        make_job(),
        size_bytes=14,
        checksum="0fbe9836cb8aa555743104c9a0553aafaef01eaa948b3ef88ab93ebdc20bd4df",
    )

    with pytest.raises(InvalidPDFSignature):
        validate_document_content(job, b"NOTPD-1.7\nbody")


class FakeStore:
    def __init__(
        self,
        job: IngestionJob | None,
        *,
        expected_lease_seconds: int = 1800,
    ) -> None:
        self.job = job
        self.expected_lease_seconds = expected_lease_seconds
        self.completed_progress: DocumentProgress | None = None
        self.failure: dict[str, object] | None = None

    async def claim_ingestion_job(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> IngestionJob | None:
        assert worker_id == "worker-1"
        assert lease_seconds == self.expected_lease_seconds
        return self.job

    async def complete_ingestion_job(self, **kwargs: object) -> bool:
        self.completed_progress = kwargs["progress"]  # type: ignore[assignment]
        return True

    async def fail_ingestion_job(self, **kwargs: object) -> str:
        self.failure = kwargs
        return "queued"


class FakeStorage:
    def __init__(self, content: bytes = b"%PDF-1.7\nbody") -> None:
        self.content = content
        self.downloaded_key: str | None = None

    async def download(self, *, storage_key: str) -> bytes:
        self.downloaded_key = storage_key
        return self.content


def make_worker(
    store: FakeStore,
    storage: FakeStorage,
    ingest: object,
) -> IngestionWorker:
    return IngestionWorker(
        store=store,
        storage=storage,
        worker_id="worker-1",
        lease_seconds=1800,
        retry_base_seconds=30,
        ingest=ingest,
    )


@pytest.mark.anyio
async def test_run_once_returns_false_when_queue_is_empty() -> None:
    storage = FakeStorage()

    def should_not_run(_document: SourceDocument, _path: Path) -> IngestionResult:
        raise AssertionError("pipeline must not run")

    worker = make_worker(FakeStore(None), storage, should_not_run)

    assert await worker.run_once() is False
    assert storage.downloaded_key is None


@pytest.mark.anyio
async def test_run_once_completes_job_with_pipeline_progress() -> None:
    store = FakeStore(make_job())
    storage = FakeStorage()
    captured_path: Path | None = None

    def successful_ingest(
        document: SourceDocument,
        path: Path,
    ) -> IngestionResult:
        nonlocal captured_path
        captured_path = path
        assert path.read_bytes() == b"%PDF-1.7\nbody"
        assert document.document_id == "document-1"
        assert document.tenant_id == "tenant-1"
        return IngestionResult(
            document_id=document.document_id,
            status=IngestionStatus.COMPLETED,
            pages_processed=10,
            pages_requiring_ocr=1,
            chunks_created=40,
            vectors_upserted=40,
            started_at=datetime(2026, 11, 3, tzinfo=timezone.utc),
            completed_at=datetime(2026, 11, 3, 0, 1, tzinfo=timezone.utc),
        )

    worker = make_worker(store, storage, successful_ingest)

    assert await worker.run_once() is True
    assert store.completed_progress == DocumentProgress(
        pages_processed=10,
        pages_requiring_ocr=1,
        chunks_created=40,
        vectors_upserted=40,
    )
    assert store.failure is None
    assert captured_path is not None
    assert not captured_path.exists()


@pytest.mark.anyio
async def test_run_once_renews_lease_while_pipeline_is_running() -> None:
    lease_renewed = Event()

    class HeartbeatStore(FakeStore):
        async def renew_ingestion_job_lease(
            self,
            *,
            job_id: str,
            worker_id: str,
        ) -> bool:
            assert job_id == "job-1"
            assert worker_id == "worker-1"
            lease_renewed.set()
            return True

    store = HeartbeatStore(make_job(), expected_lease_seconds=1)

    def wait_for_heartbeat(
        document: SourceDocument,
        _path: Path,
    ) -> IngestionResult:
        assert lease_renewed.wait(timeout=2), "lease was not renewed"
        return IngestionResult(
            document_id=document.document_id,
            status=IngestionStatus.COMPLETED,
            vectors_upserted=1,
        )

    worker = IngestionWorker(
        store=store,
        storage=FakeStorage(),
        worker_id="worker-1",
        lease_seconds=1,
        retry_base_seconds=30,
        ingest=wait_for_heartbeat,
    )

    assert await worker.run_once() is True
    assert store.completed_progress == DocumentProgress(vectors_upserted=1)
    assert store.failure is None


@pytest.mark.anyio
async def test_run_once_records_pipeline_failure_for_retry() -> None:
    store = FakeStore(make_job())

    def failed_ingest(
        document: SourceDocument,
        _path: Path,
    ) -> IngestionResult:
        return IngestionResult(
            document_id=document.document_id,
            status=IngestionStatus.FAILED,
            pages_processed=2,
            errors=["InvalidPDFError: PDF could not be parsed"],
        )

    worker = make_worker(store, FakeStorage(), failed_ingest)

    assert await worker.run_once() is True
    assert store.completed_progress is None
    assert store.failure is not None
    assert store.failure["error_code"] == "InvalidPDFError"
    assert store.failure["error_message"] == "PDF could not be parsed"
    assert store.failure["progress"] == DocumentProgress(pages_processed=2)


@pytest.mark.anyio
async def test_run_once_converts_unexpected_error_to_safe_failure() -> None:
    store = FakeStore(make_job())

    class FailingStorage(FakeStorage):
        async def download(self, *, storage_key: str) -> bytes:
            raise RuntimeError("provider response included sensitive details")

    def should_not_run(_document: SourceDocument, _path: Path) -> IngestionResult:
        raise AssertionError("pipeline must not run")

    worker = make_worker(store, FailingStorage(), should_not_run)

    assert await worker.run_once() is True
    assert store.failure is not None
    assert store.failure["error_code"] == "RuntimeError"
    assert store.failure["error_message"] == "Unexpected ingestion worker failure."
    assert "sensitive" not in str(store.failure)
