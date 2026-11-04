from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from integrations.nvidia_nim import (
    NvidiaNIMEmbeddings,
    _truncate_and_normalize,
)


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
