from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WebSearchState(TypedDict, total=False):
    """State passed between the trusted web-search graph nodes."""

    query: str
    messages: Annotated[list[AnyMessage], add_messages]
