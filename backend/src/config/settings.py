from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_api_key: str
    embedding_model: str
    embedding_dimension: int = 768
    chat_model: str
    vision_model: str

    tavily_api_key: str

    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_namespace: str

    ingestion_api_key: SecretStr
    api_key_tenant_id: str
    api_key_id: str = "primary"


@lru_cache
def get_settings() -> Settings:
    return Settings()
