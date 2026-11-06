from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from integrations.nvidia_nim import (
    NvidiaNIMEmbeddings,
    _truncate_and_normalize,
    get_nvidia_nim_chat_model,
)


@pytest.fixture(autouse=True)
def clear_chat_model_cache() -> Iterator[None]:
    get_nvidia_nim_chat_model.cache_clear()
    yield
    get_nvidia_nim_chat_model.cache_clear()


def test_truncate_and_normalize_returns_requested_dimension() -> None:
    vector = _truncate_and_normalize([3.0, 4.0, 12.0], dimension=2)

    assert vector == pytest.approx([0.6, 0.8])


def test_truncate_and_normalize_rejects_short_vector() -> None:
    with pytest.raises(ValueError, match="expected at least 3"):
        _truncate_and_normalize([1.0, 2.0], dimension=3)


def test_documents_use_passage_mode_and_queries_use_query_mode() -> None:
    client = MagicMock()
    client.create.return_value = {
        "data": [{"embedding": [3.0, 4.0, 12.0]}]
    }
    embeddings = NvidiaNIMEmbeddings(
        client=client,
        model="nvidia/nemotron-3.5-embed-1b",
        api_key=SecretStr("test-key"),
        base_url="https://integrate.api.nvidia.com/v1",
        output_dimension=2,
        check_embedding_ctx_length=False,
        tiktoken_enabled=False,
        max_retries=0,
    )

    document_vector = embeddings.embed_documents(["medical document"])[0]
    document_body = client.create.call_args.kwargs["extra_body"]
    query_vector = embeddings.embed_query("medical question")
    query_body = client.create.call_args.kwargs["extra_body"]

    assert document_vector == pytest.approx([0.6, 0.8])
    assert query_vector == pytest.approx([0.6, 0.8])
    assert document_body == {"input_type": "passage", "truncate": "END"}
    assert query_body == {"input_type": "query", "truncate": "END"}


def test_get_nvidia_nim_chat_model_unconfigured_raises() -> None:
    mock_settings = SimpleNamespace(
        nvidia_api_key=None,
        nvidia_chat_model=None,
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
    )

    with patch("integrations.nvidia_nim.get_settings", return_value=mock_settings):
        with pytest.raises(ValueError, match="NVIDIA API key not configured"):
            get_nvidia_nim_chat_model()

    mock_settings.nvidia_api_key = SecretStr("key")
    mock_settings.nvidia_chat_model = ""
    with patch("integrations.nvidia_nim.get_settings", return_value=mock_settings):
        get_nvidia_nim_chat_model.cache_clear()
        with pytest.raises(ValueError, match="NVIDIA chat model not configured"):
            get_nvidia_nim_chat_model()


def test_get_nvidia_nim_chat_model_success() -> None:
    mock_settings = SimpleNamespace(
        nvidia_api_key=SecretStr("valid-key"),
        nvidia_chat_model="meta/llama-3.1-70b-instruct",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
    )

    with patch("integrations.nvidia_nim.get_settings", return_value=mock_settings):
        model_1 = get_nvidia_nim_chat_model()
        model_2 = get_nvidia_nim_chat_model()

        assert isinstance(model_1, ChatOpenAI)
        assert model_1 is model_2
        assert model_1.model_name == "meta/llama-3.1-70b-instruct"
        assert model_1.temperature == 0.1
        assert model_1.max_retries == 2
        assert model_1.request_timeout == 60
