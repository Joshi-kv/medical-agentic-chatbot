from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_api_key: str | None = None
    chat_model: str | None = None
    vision_model: str | None = None

    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    embedding_model: str = "nvidia/nemotron-3.5-embed-1b"
    embedding_dimension: int = Field(default=1024, gt=0, le=2048)
    embedding_batch_size: int = Field(default=32, gt=0, le=128)
    embedding_timeout_seconds: float = Field(default=120, gt=0, le=600)

    tavily_api_key: str

    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str

    ingestion_api_key: SecretStr
    api_key_tenant_id: str
    api_key_id: str = "primary"

    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_document_bucket: str = "medical-documents"
    max_upload_bytes: int = 20 * 1024 * 1024
    ingestion_pipeline_version: str = "1"
    ingestion_worker_poll_seconds: float = Field(default=5, gt=0, le=300)
    ingestion_job_lease_seconds: int = Field(default=1800, gt=0, le=86400)
    ingestion_retry_base_seconds: int = Field(default=30, gt=0, le=3600)


@lru_cache
def get_settings() -> Settings:
    return Settings()
