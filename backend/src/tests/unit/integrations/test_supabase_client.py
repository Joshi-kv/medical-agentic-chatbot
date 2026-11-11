from typing import Any

from pydantic import SecretStr

from config.settings import Settings
from integrations.supabase import build_document_service
from services.documents import DefaultDocumentService


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "embedding_model": "embedding-model",
        "tavily_api_key": "tavily",
        "pinecone_api_key": "pinecone",
        "pinecone_index_name": "index",
        "pinecone_namespace": "namespace",
        "ingestion_api_key": "ingestion-key",
        "api_key_tenant_id": "tenant-1",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_build_document_service_returns_none_without_supabase_configuration() -> None:
    assert build_document_service(make_settings()) is None


def test_build_document_service_uses_service_role_credential() -> None:
    captured: tuple[str, str] | None = None

    class Client:
        class Storage:
            def from_(self, _bucket: str) -> object:
                return object()

        storage = Storage()

    def client_factory(url: str, key: str) -> Client:
        nonlocal captured
        captured = (url, key)
        return Client()

    settings = make_settings(
        supabase_url="https://project.supabase.co",
        supabase_service_role_key=SecretStr("service-role-secret"),
    )

    service = build_document_service(settings, client_factory=client_factory)

    assert isinstance(service, DefaultDocumentService)
    assert captured == ("https://project.supabase.co", "service-role-secret")
