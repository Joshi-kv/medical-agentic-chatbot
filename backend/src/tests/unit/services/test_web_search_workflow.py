import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import ToolMessage

from services.web_search_workflow import WebSearchWorkflow, WebSearchWorkflowError


@pytest.mark.anyio
async def test_workflow_returns_tool_evidence() -> None:
    graph = AsyncMock()
    graph.ainvoke.return_value = {
        "messages": [
            ToolMessage(
                content=json.dumps(
                    {
                        "items": [
                            {
                                "citation_id": "C1",
                                "chunk_id": "web-1",
                                "text": "Trusted medical evidence.",
                                "document_id": "web",
                                "page_number": 0,
                                "source": "https://www.cdc.gov/health",
                                "score": 0.9,
                                "title": "CDC Health",
                            }
                        ]
                    }
                ),
                tool_call_id="call-1",
            )
        ]
    }
    workflow = WebSearchWorkflow(graph)

    result = await workflow.search("fever")

    assert result.sufficient is True
    assert result.items[0].source == "https://www.cdc.gov/health"


@pytest.mark.anyio
async def test_workflow_requires_tool_message() -> None:
    graph = AsyncMock()
    graph.ainvoke.return_value = {"messages": []}
    workflow = WebSearchWorkflow(graph)

    with pytest.raises(WebSearchWorkflowError, match="did not call"):
        await workflow.search("fever")
