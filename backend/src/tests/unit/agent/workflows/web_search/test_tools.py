import json
from unittest.mock import AsyncMock

import pytest

from agent.workflows.web_search.tools import create_medical_web_search_tool
from services.evidence import Evidence, EvidencePackage


@pytest.mark.anyio
async def test_search_tool_serializes_evidence_package() -> None:
    service = AsyncMock()
    service.collect_evidence.return_value = EvidencePackage(
        query="fever",
        sufficient=True,
        items=[
            Evidence(
                citation_id="C1",
                chunk_id="web-1",
                text="Trusted evidence",
                document_id="web",
                page_number=0,
                source="https://www.cdc.gov/fever",
                score=0.9,
                title="CDC",
            )
        ],
    )
    search_tool = create_medical_web_search_tool(service)

    raw_result = await search_tool.ainvoke({"query": "fever"})
    result = json.loads(raw_result)

    assert result["sufficient"] is True
    assert result["items"][0]["document_id"] == "web"
    service.collect_evidence.assert_awaited_once_with("fever")
