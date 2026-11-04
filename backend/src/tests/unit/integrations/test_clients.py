import time
from uuid import uuid4

import pytest

from integrations.nvidia_nim import get_nvidia_nim_embeddings
from integrations.pinecone import get_pinecone_index
from integrations.tavily import get_tavily_client

pytestmark = pytest.mark.integration


def test_nvidia_embeddings_returns_expected_dimension() -> None:
    embeddings = get_nvidia_nim_embeddings()

    vector = embeddings.embed_query("The heart pumps blood through the body.")

    assert len(vector) == 1024
    assert all(isinstance(value, float) for value in vector)


def test_pinecone_upsert_query_and_delete() -> None:
    embeddings = get_nvidia_nim_embeddings()
    index = get_pinecone_index()

    namespace = "integration-tests"
    vector_id = f"test-{uuid4()}"
    vector = embeddings.embed_query("Aspirin is an antiplatelet medication.")

    try:
        index.upsert(
            namespace=namespace,
            vectors=[
                {
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "source": "pytest",
                        "temporary": True,
                    },
                }
            ],
        )

        matches = []

        # Pinecone writes may not become searchable immediately.
        for _ in range(5):
            response = index.query(
                namespace=namespace,
                vector=vector,
                top_k=1,
                include_metadata=True,
            )
            matches = response.matches

            if matches and matches[0].id == vector_id:
                break

            time.sleep(1)

        assert matches
        assert matches[0].id == vector_id
        assert matches[0].metadata["source"] == "pytest"
    finally:
        index.delete(namespace=namespace, ids=[vector_id])


def test_tavily_search_returns_results() -> None:
    client = get_tavily_client()

    response = client.search(
        "World Health Organization official website",
        max_results=1,
        include_answer=False,
    )

    assert response["results"]
    assert response["results"][0]["url"]
