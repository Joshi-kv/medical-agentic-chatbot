from unittest.mock import MagicMock

import pytest

from services.web_search import WebSearchError, WebSearchService


@pytest.mark.anyio
async def test_collect_evidence_keeps_only_trusted_https_results() -> None:
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "url": "https://www.cdc.gov/fever/about/index.html",
                "title": "About fever",
                "content": "Fever is a temporary rise in body temperature. " * 4,
                "score": 0.91,
            },
            {
                "url": "https://untrusted.example/fever",
                "title": "Untrusted",
                "content": "This content is long enough but must not be accepted. " * 4,
                "score": 0.99,
            },
        ]
    }
    service = WebSearchService(client, ["cdc.gov"], max_results=3)

    result = await service.collect_evidence("What is fever?")

    assert result.sufficient is True
    assert len(result.items) == 1
    assert result.items[0].document_id == "web"
    assert result.items[0].source.startswith("https://www.cdc.gov/")
    client.search.assert_called_once_with(
        query="What is fever? medical patient information",
        topic="general",
        search_depth="basic",
        max_results=3,
        include_domains=["cdc.gov"],
        include_answer=False,
        include_raw_content=False,
    )


@pytest.mark.anyio
async def test_collect_evidence_rejects_results_outside_allowlist() -> None:
    client = MagicMock()
    client.search.return_value = {
        "results": [
            {
                "url": "https://not-cdc.gov/fever",
                "title": "Lookalike domain",
                "content": (
                    "A result from a domain that only resembles the allowlist. " * 3
                ),
            }
        ]
    }
    service = WebSearchService(client, ["cdc.gov"])

    with pytest.raises(WebSearchError, match="No relevant evidence"):
        await service.collect_evidence("fever")


def test_web_search_service_rejects_url_in_domain_configuration() -> None:
    with pytest.raises(ValueError, match="Invalid trusted domain"):
        WebSearchService(MagicMock(), ["https://cdc.gov/health"])
