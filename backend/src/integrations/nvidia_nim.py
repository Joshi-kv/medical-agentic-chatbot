import math
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import get_settings


class NvidiaNIMEmbeddings(OpenAIEmbeddings):
    """OpenAI-compatible NVIDIA NIM embeddings with retrieval-aware inputs."""

    output_dimension: int

    def embed_documents(
        self,
        texts: list[str],
        chunk_size: int | None = None,
        **kwargs: object,
    ) -> list[list[float]]:
        extra_body = {
            "input_type": "passage",
            "truncate": "END",
            **dict(kwargs.pop("extra_body", {})),
        }
        vectors = super().embed_documents(
            texts,
            chunk_size=chunk_size,
            extra_body=extra_body,
            **kwargs,
        )
        return [
            _truncate_and_normalize(vector, self.output_dimension) for vector in vectors
        ]

    def embed_query(self, text: str, **kwargs: object) -> list[float]:
        extra_body = {
            "input_type": "query",
            "truncate": "END",
            **dict(kwargs.pop("extra_body", {})),
        }
        vectors = OpenAIEmbeddings.embed_documents(
            self,
            [text],
            extra_body=extra_body,
            **kwargs,
        )
        return _truncate_and_normalize(vectors[0], self.output_dimension)


def _truncate_and_normalize(vector: list[float], dimension: int) -> list[float]:
    if dimension <= 0:
        raise ValueError("Embedding dimension must be positive.")
    if len(vector) < dimension:
        raise ValueError(
            f"NVIDIA returned {len(vector)} dimensions; expected at least {dimension}."
        )

    truncated = vector[:dimension]
    magnitude = math.sqrt(sum(value * value for value in truncated))
    if magnitude == 0:
        raise ValueError("NVIDIA returned a zero-magnitude embedding.")
    return [value / magnitude for value in truncated]


@lru_cache()
def get_nvidia_nim_embeddings() -> NvidiaNIMEmbeddings:
    settings = get_settings()
    if (
        settings.nvidia_api_key is None
        or not settings.nvidia_api_key.get_secret_value()
    ):
        raise ValueError("NVIDIA API key not configured.")
    if not settings.embedding_model:
        raise ValueError("Embedding model not configured.")

    return NvidiaNIMEmbeddings(
        model=settings.embedding_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        output_dimension=settings.embedding_dimension,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
        chunk_size=settings.embedding_batch_size,
        max_retries=0,
        timeout=settings.embedding_timeout_seconds,
    )


@lru_cache()
def get_nvidia_nim_chat_model() -> ChatOpenAI:
    settings = get_settings()
    if (
        settings.nvidia_api_key is None
        or not settings.nvidia_api_key.get_secret_value()
    ):
        raise ValueError("NVIDIA API key not configured.")
    if not settings.nvidia_chat_model:
        raise ValueError("NVIDIA chat model not configured.")

    return ChatOpenAI(
        model=settings.nvidia_chat_model,
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        temperature=0.1,
        max_tokens=400,
        max_retries=0,
        timeout=30,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )
