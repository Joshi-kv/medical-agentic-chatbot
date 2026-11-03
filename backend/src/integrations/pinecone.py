from functools import lru_cache

from pinecone import Index, Pinecone, ServerlessSpec

from config.settings import get_settings


@lru_cache()
def get_pinecone_client() -> Pinecone:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise ValueError("Pinecone API key not configured.")
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache()
def get_pinecone_index() -> Index:
    settings = get_settings()
    if not settings.pinecone_index_name:
        raise ValueError("Pinecone index name not configured.")
    return get_pinecone_client().index(name=settings.pinecone_index_name)


@lru_cache()
def create_pinecone_index() -> Index:
    settings = get_settings()
    if not settings.pinecone_index_name:
        raise ValueError("Pinecone index name not configured.")

    client = get_pinecone_client()
    index_names = {index.name for index in client.indexes.list()}

    if settings.pinecone_index_name not in index_names:
        client.indexes.create(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    return client.index(name=settings.pinecone_index_name)
