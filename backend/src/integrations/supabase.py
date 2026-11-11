from collections.abc import Callable
from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from config.settings import Settings, get_settings
from repositories.supabase_documents import SupabaseDocumentStore
from services.documents import DefaultDocumentService
from storage.supabase_documents import SupabaseDocumentStorage


@lru_cache
def get_supabase_client() -> Client:
    settings = get_settings()

    if settings.supabase_url is None or settings.supabase_service_role_key is None:
        raise ValueError("Supabase URL and Service Role Key must be set")

    return create_client(
        settings.supabase_url,
        settings.supabase_service_role_key.get_secret_value(),
    )


def build_document_service(
    settings: Settings,
    *,
    client_factory: Callable[[str, str], Any] = create_client,
) -> DefaultDocumentService | None:
    if settings.supabase_url is None or settings.supabase_service_role_key is None:
        return None

    client = client_factory(
        settings.supabase_url,
        settings.supabase_service_role_key.get_secret_value(),
    )
    return DefaultDocumentService(
        storage=SupabaseDocumentStorage(
            client=client,
            bucket=settings.supabase_document_bucket,
        ),
        store=SupabaseDocumentStore(client=client),
        max_upload_bytes=settings.max_upload_bytes,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        pipeline_version=settings.ingestion_pipeline_version,
    )
