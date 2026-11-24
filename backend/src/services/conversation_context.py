import asyncio
import json
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from prompts.conversation_context import (
    CONTEXTUALIZE_SYSTEM_PROMPT,
    CONTEXTUALIZE_USER_PROMPT,
)

CONTEXTUALIZATION_TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class ContextMessage:
    role: Literal["user", "assistant"]
    content: str


class ContextualizedQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class ConversationContextError(Exception):
    pass


class ConversationContextualizer:
    """Rewrite a contextual follow-up as a standalone search query."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model
        self._parser = PydanticOutputParser(
            pydantic_object=ContextualizedQuery,
        )

    async def rewrite(
        self,
        query: str,
        history: list[ContextMessage],
    ) -> str:
        normalized_query = query.strip()

        if not normalized_query:
            raise ConversationContextError("Query cannot be empty.")

        if not history:
            return normalized_query

        serialized_history = json.dumps(
            [
                {"role": message.role, "content": message.content}
                for message in history
            ],
            ensure_ascii=False,
        )
        prompt = CONTEXTUALIZE_USER_PROMPT.format(
            history=serialized_history,
            query=normalized_query,
            format_instructions=self._parser.get_format_instructions(),
        )

        try:
            async with asyncio.timeout(CONTEXTUALIZATION_TIMEOUT_SECONDS):
                response = await self._chat_model.ainvoke(
                    [
                        SystemMessage(content=CONTEXTUALIZE_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
        except TimeoutError as error:
            raise ConversationContextError(
                "Conversation contextualization timed out."
            ) from error
        except Exception as error:
            raise ConversationContextError(
                "Conversation contextualization failed."
            ) from error

        if not isinstance(response.content, str):
            raise ConversationContextError(
                "Contextualizer returned unsupported content."
            )

        try:
            result = self._parser.parse(response.content)
        except Exception as error:
            raise ConversationContextError(
                "Contextualizer returned invalid output."
            ) from error

        return result.query.strip()
