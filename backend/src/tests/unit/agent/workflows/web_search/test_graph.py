from unittest.mock import AsyncMock

from langchain_core.tools import tool

from agent.workflows.web_search.graph import build_web_search_graph


@tool("search_trusted_medical_web")
async def fake_search(query: str) -> str:
    """Return fake trusted evidence."""
    return query


def test_build_web_search_graph_contains_expected_nodes() -> None:
    nodes = AsyncMock()

    graph = build_web_search_graph(nodes=nodes, search_tool=fake_search)
    graph_nodes = set(graph.get_graph().nodes)

    assert graph_nodes == {"__start__", "request_web_search", "tools", "__end__"}
