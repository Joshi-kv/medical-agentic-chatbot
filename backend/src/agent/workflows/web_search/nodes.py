from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from agent.workflows.web_search.state import WebSearchState
from prompts.web_search import TOOL_SELECTION_PROMPT

WebSearchNextStep = Literal["tools", "no_tool"]


class WebSearchNodes:
    """Nodes that ask the chat model to call the trusted search tool."""

    def __init__(self, chat_model: BaseChatModel, search_tool: BaseTool) -> None:
        self._tool_model = chat_model.bind_tools([search_tool], tool_choice="required")

    async def request_web_search(
        self,
        state: WebSearchState,
    ) -> dict[str, list[AnyMessage]]:
        query = state.get("query")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Web-search query is missing or invalid.")

        response = await self._tool_model.ainvoke(
            [
                SystemMessage(content=TOOL_SELECTION_PROMPT),
                HumanMessage(content=query.strip()),
            ]
        )

        return {"messages": [response]}

    def choose_after_model(
        self,
        state: WebSearchState,
    ) -> WebSearchNextStep:
        messages = state.get("messages", [])

        if not messages:
            return "no_tool"

        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        return "no_tool"
