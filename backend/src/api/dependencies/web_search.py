from functools import lru_cache

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agent.workflows.web_search.graph import (
    build_web_search_graph,
)
from agent.workflows.web_search.nodes import WebSearchNodes
from agent.workflows.web_search.tools import (
    create_medical_web_search_tool,
)
from config.settings import get_settings
from integrations.nvidia_nim import (
    get_nvidia_nim_chat_model,
)
from integrations.tavily import get_tavily_client
from services.web_search import WebSearchService
from services.web_search_workflow import WebSearchWorkflow


@lru_cache
def get_web_search_service() -> WebSearchService:
    settings = get_settings()

    return WebSearchService(
        client=get_tavily_client(),
        trusted_domains=settings.tavily_trusted_domains,
        max_results=settings.tavily_max_results,
        search_depth=settings.tavily_search_depth,
    )


@lru_cache
def get_web_search_tool() -> BaseTool:
    return create_medical_web_search_tool(
        service=get_web_search_service(),
    )


@lru_cache
def get_web_search_nodes() -> WebSearchNodes:
    return WebSearchNodes(
        chat_model=get_nvidia_nim_chat_model(),
        search_tool=get_web_search_tool(),
    )


@lru_cache
def get_web_search_graph() -> CompiledStateGraph:
    return build_web_search_graph(
        nodes=get_web_search_nodes(),
        search_tool=get_web_search_tool(),
    )


@lru_cache
def get_web_search_workflow() -> WebSearchWorkflow:
    return WebSearchWorkflow(
        graph=get_web_search_graph(),
    )


def get_optional_web_search_workflow() -> WebSearchWorkflow | None:
    settings = get_settings()

    if settings.tavily_api_key is None:
        return None

    if not settings.tavily_api_key.get_secret_value():
        return None

    return get_web_search_workflow()
