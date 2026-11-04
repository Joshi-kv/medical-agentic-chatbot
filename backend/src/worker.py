import asyncio
import logging
import os
import socket
from collections.abc import Awaitable, Callable
from typing import Any

from anyio import sleep as anyio_sleep
from supabase import create_client

from config.settings import Settings, get_settings
from ingestion.pipeline import ingest_pdf
from repositories.supabase_documents import SupabaseDocumentStore
from storage.supabase_documents import SupabaseDocumentStorage
from worker_runtime import IngestionWorker

logger = logging.getLogger(__name__)


class WorkerConfigurationError(RuntimeError):
    pass


def build_ingestion_worker(
    settings: Settings,
    *,
    client_factory: Callable[[str, str], Any] = create_client,
    worker_id: str | None = None,
) -> IngestionWorker:
    if settings.supabase_url is None or settings.supabase_service_role_key is None:
        raise WorkerConfigurationError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required"
        )
    client = client_factory(
        settings.supabase_url,
        settings.supabase_service_role_key.get_secret_value(),
    )
    return IngestionWorker(
        store=SupabaseDocumentStore(client=client),
        storage=SupabaseDocumentStorage(
            client=client,
            bucket=settings.supabase_document_bucket,
        ),
        worker_id=worker_id or _default_worker_id(),
        lease_seconds=settings.ingestion_job_lease_seconds,
        retry_base_seconds=settings.ingestion_retry_base_seconds,
        ingest=ingest_pdf,
    )


async def poll_worker(
    *,
    worker: Any,
    poll_seconds: float,
    sleep: Callable[[float], Awaitable[None]] = anyio_sleep,
) -> None:
    while True:
        try:
            processed = await worker.run_once()
        except Exception as error:
            logger.error(
                "ingestion_worker_iteration_failed",
                extra={"error_type": type(error).__name__},
            )
            await sleep(poll_seconds)
            continue

        if not processed:
            await sleep(poll_seconds)


async def run_worker_async(settings: Settings | None = None) -> None:
    resolved_settings = settings or get_settings()
    worker = build_ingestion_worker(resolved_settings)
    logger.info("ingestion_worker_started")
    await poll_worker(
        worker=worker,
        poll_seconds=resolved_settings.ingestion_worker_poll_seconds,
    )


def run_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(run_worker_async())


def _default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


if __name__ == "__main__":
    run_worker()
