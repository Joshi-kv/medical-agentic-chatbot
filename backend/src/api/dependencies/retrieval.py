from functools import lru_cache

from config.settings import get_settings
from integrations.nvidia_nim import get_nvidia_nim_embeddings
from integrations.pinecone import get_pinecone_index
from services.retrieval import RetrievalService


@lru_cache
def get_retrieval_service() -> RetrievalService:
    settings = get_settings()

    return RetrievalService(
        embeddings=get_nvidia_nim_embeddings(),
        index=get_pinecone_index(),
        namespace=settings.pinecone_namespace,
    )
