from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from agent.workflows.web_search.nodes import WebSearchNodes


@pytest.mark.anyio
async def test_request_web_search_requires_tool_call() -> None:
    bound_model = AsyncMock()
    response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_trusted_medical_web",
                "args": {"query": "fever patient information"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    bound_model.ainvoke.return_value = response
    chat_model = MagicMock()
    chat_model.bind_tools.return_value = bound_model
    search_tool = MagicMock()
    nodes = WebSearchNodes(chat_model, search_tool)

    result = await nodes.request_web_search({"query": "What is fever?"})

    chat_model.bind_tools.assert_called_once_with(
        [search_tool],
        tool_choice="required",
    )
    assert result == {"messages": [response]}
    assert nodes.choose_after_model(result) == "tools"


def test_choose_after_model_stops_without_tool_call() -> None:
    chat_model = MagicMock()
    chat_model.bind_tools.return_value = AsyncMock()
    nodes = WebSearchNodes(chat_model, MagicMock())

    assert nodes.choose_after_model({"messages": []}) == "no_tool"
    assert (
        nodes.choose_after_model({"messages": [AIMessage(content="No tool")]})
        == "no_tool"
    )
