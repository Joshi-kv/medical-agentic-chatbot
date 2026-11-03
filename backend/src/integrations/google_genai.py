from functools import lru_cache

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from config.settings import get_settings


@lru_cache()
def get_google_genai_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    if not settings.google_api_key or not settings.embedding_model:
        raise ValueError("Google API key and embedding model not configured.")
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimension,
    )


@lru_cache()
def get_google_chat_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.google_api_key or not settings.chat_model:
        raise ValueError("Google API key and chat model not configured.")
    return ChatGoogleGenerativeAI(
        model=settings.chat_model, api_key=settings.google_api_key
    )
