from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    google_api_key: str
    embedding_model: str
    chat_model: str
    vision_model: str

    tavily_api_key: str

    pinecone_api_key: str
    pinecone_index_name: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
