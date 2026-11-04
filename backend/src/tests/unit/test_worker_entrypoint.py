import asyncio
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from config.settings import Settings
from worker import (
    WorkerConfigurationError,
    build_ingestion_worker,
    poll_worker,
)
from worker_runtime import IngestionWorker


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "google_api_key": "google",
        "embedding_model": "embedding-model",
        "chat_model": "chat-model",
        "vision_model": "vision-model",
        "tavily_api_key": "tavily",
        "pinecone_api_key": "pinecone",
        "pinecone_index_name": "index",
        "pinecone_namespace": "namespace",
        "ingestion_api_key": "ingestion-key",
        "api_key_tenant_id": "tenant-1",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_build_worker_requires_supabase_configuration() -> None:
    with pytest.raises(WorkerConfigurationError):
        build_ingestion_worker(make_settings())


def test_build_worker_uses_service_role_without_exposing_it() -> None:
    captured: tuple[str, str] | None = None

    class StorageRoot:
        def from_(self, _bucket: str) -> object:
            return object()

    class Client:
        storage = StorageRoot()

    def client_factory(url: str, key: str) -> Client:
        nonlocal captured
        captured = (url, key)
        return Client()

    settings = make_settings(
        supabase_url="https://project.supabase.co",
        supabase_service_role_key=SecretStr("service-role-secret"),
    )

    worker = build_ingestion_worker(
        settings,
        client_factory=client_factory,
        worker_id="worker-1",
    )

    assert isinstance(worker, IngestionWorker)
    assert captured == ("https://project.supabase.co", "service-role-secret")


def test_worker_poll_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        make_settings(ingestion_worker_poll_seconds=0)


@pytest.mark.anyio
async def test_poll_worker_drains_jobs_then_sleeps_when_idle() -> None:
    class Worker:
        def __init__(self) -> None:
            self.results = iter([True, False])
            self.run_count = 0

        async def run_once(self) -> bool:
            self.run_count += 1
            return next(self.results)

    worker = Worker()
    sleep_calls: list[float] = []

    async def cancelling_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await poll_worker(
            worker=worker,
            poll_seconds=7,
            sleep=cancelling_sleep,
        )

    assert worker.run_count == 2
    assert sleep_calls == [7]
