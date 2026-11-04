from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from anyio import create_task_group, sleep, to_thread

from ingestion.schemas import IngestionResult, IngestionStatus, SourceDocument
from repositories.supabase_documents import IngestionJob
from services.documents import DocumentProgress


class DocumentIntegrityError(Exception):
    pass


class DocumentSizeMismatch(DocumentIntegrityError):
    pass


class DocumentChecksumMismatch(DocumentIntegrityError):
    pass


class InvalidPDFSignature(DocumentIntegrityError):
    pass


class IngestionLeaseLost(RuntimeError):
    pass


def validate_document_content(job: IngestionJob, content: bytes) -> None:
    if len(content) != job.size_bytes:
        raise DocumentSizeMismatch(
            f"Expected {job.size_bytes} bytes, received {len(content)}"
        )
    if sha256(content).hexdigest() != job.checksum:
        raise DocumentChecksumMismatch("Document checksum does not match")
    if not content.startswith(b"%PDF-"):
        raise InvalidPDFSignature("Document is missing the PDF signature")


class IngestionWorker:
    def __init__(
        self,
        *,
        store: Any,
        storage: Any,
        worker_id: str,
        lease_seconds: int,
        retry_base_seconds: int,
        ingest: Callable[[SourceDocument, Path], IngestionResult],
    ) -> None:
        self._store = store
        self._storage = storage
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._lease_heartbeat_seconds = max(0.1, lease_seconds / 3)
        self._retry_base_seconds = retry_base_seconds
        self._ingest = ingest

    async def run_once(self) -> bool:
        job = await self._store.claim_ingestion_job(
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if job is None:
            return False

        try:
            content = await self._storage.download(storage_key=job.storage_key)
            validate_document_content(job, content)
            document = SourceDocument(
                document_id=job.document_id,
                filename=job.filename,
                content_type="application/pdf",
                size_bytes=job.size_bytes,
                checksum=job.checksum,
                title=job.title,
                source=job.storage_key,
                tenant_id=job.tenant_id,
                metadata={"ingestion_attempt": job.attempts},
            )

            with TemporaryDirectory(prefix="medical-ingestion-") as directory:
                pdf_path = Path(directory) / "document.pdf"
                await to_thread.run_sync(pdf_path.write_bytes, content)
                result = await self._run_ingestion_with_heartbeat(
                    document,
                    job=job,
                    pdf_path=pdf_path,
                )

            progress = _progress_from_result(result)
            if result.status == IngestionStatus.COMPLETED:
                await self._store.complete_ingestion_job(
                    job_id=job.job_id,
                    document_id=job.document_id,
                    tenant_id=job.tenant_id,
                    worker_id=self._worker_id,
                    progress=progress,
                )
            else:
                error_code, error_message = _pipeline_error(result)
                await self._record_failure(
                    job=job,
                    error_code=error_code,
                    error_message=error_message,
                    progress=progress,
                )
        except DocumentIntegrityError as error:
            await self._record_failure(
                job=job,
                error_code=type(error).__name__,
                error_message=str(error),
                progress=DocumentProgress(),
            )
        except Exception as error:
            await self._record_failure(
                job=job,
                error_code=type(error).__name__,
                error_message="Unexpected ingestion worker failure.",
                progress=DocumentProgress(),
            )

        return True

    async def _run_ingestion_with_heartbeat(
        self,
        document: SourceDocument,
        *,
        job: IngestionJob,
        pdf_path: Path,
    ) -> IngestionResult:
        result: IngestionResult | None = None

        async with create_task_group() as task_group:
            async def run_pipeline() -> None:
                nonlocal result
                result = await to_thread.run_sync(
                    self._ingest,
                    document,
                    pdf_path,
                )
                task_group.cancel_scope.cancel()

            async def renew_lease() -> None:
                while True:
                    await sleep(self._lease_heartbeat_seconds)
                    renewed = await self._store.renew_ingestion_job_lease(
                        job_id=job.job_id,
                        worker_id=self._worker_id,
                    )
                    if not renewed:
                        raise IngestionLeaseLost(
                            "Ingestion job lease is no longer owned by this worker."
                        )

            task_group.start_soon(run_pipeline)
            task_group.start_soon(renew_lease)

        if result is None:
            raise IngestionLeaseLost("Ingestion stopped before producing a result.")
        return result

    async def _record_failure(
        self,
        *,
        job: IngestionJob,
        error_code: str,
        error_message: str,
        progress: DocumentProgress,
    ) -> None:
        await self._store.fail_ingestion_job(
            job_id=job.job_id,
            document_id=job.document_id,
            tenant_id=job.tenant_id,
            worker_id=self._worker_id,
            error_code=error_code,
            error_message=error_message,
            progress=progress,
            retry_base_seconds=self._retry_base_seconds,
        )


def _progress_from_result(result: IngestionResult) -> DocumentProgress:
    return DocumentProgress(
        pages_processed=result.pages_processed,
        pages_requiring_ocr=result.pages_requiring_ocr,
        chunks_created=result.chunks_created,
        vectors_upserted=result.vectors_upserted,
    )


def _pipeline_error(result: IngestionResult) -> tuple[str, str]:
    if not result.errors:
        return "IngestionFailed", "The ingestion pipeline failed."
    error = result.errors[0]
    error_code, separator, message = error.partition(": ")
    if not separator:
        return "IngestionFailed", error[:1000]
    return error_code[:100], message[:1000]
