from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.workflows.web_search.nodes import WebSearchNodes
from agent.workflows.web_search.state import WebSearchState


def build_web_search_graph(
    nodes: WebSearchNodes,
    search_tool: BaseTool,
) -> CompiledStateGraph:
    """Build the one-shot trusted medical web-search workflow."""

    builder = StateGraph(WebSearchState)

    builder.add_node("request_web_search", nodes.request_web_search)
    builder.add_node(
        "tools",
        ToolNode(
            tools=[search_tool],
            handle_tool_errors=("Trusted medical web search service is unavailable"),
        ),
    )

    builder.add_edge(
        START,
        "request_web_search",
    )
    builder.add_conditional_edges(
        "request_web_search",
        nodes.choose_after_model,
        {"tools": "tools", "no_tool": END},
    )
    builder.add_edge("tools", END)

    return builder.compile(name="trusted_medical_web_search")
