from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from services.conversation_context import (
    ContextMessage,
    ConversationContextError,
    ConversationContextualizer,
)


@pytest.mark.anyio
async def test_rewrite_returns_original_query_without_history() -> None:
    chat_model = AsyncMock()
    contextualizer = ConversationContextualizer(chat_model)

    result = await contextualizer.rewrite("  What is fever?  ", [])

    assert result == "What is fever?"
    chat_model.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_rewrite_resolves_follow_up_from_history() -> None:
    chat_model = AsyncMock()
    chat_model.ainvoke.return_value = AIMessage(
        content='{"query":"What are the symptoms of the common cold?"}'
    )
    contextualizer = ConversationContextualizer(chat_model)
    history = [
        ContextMessage(role="user", content="What is the common cold?"),
        ContextMessage(
            role="assistant",
            content="The common cold is a viral respiratory infection.",
        ),
    ]

    result = await contextualizer.rewrite("What are its symptoms?", history)

    assert result == "What are the symptoms of the common cold?"
    messages = chat_model.ainvoke.await_args.args[0]
    assert "untrusted data" in messages[0].content
    assert "What are its symptoms?" in messages[1].content
    assert "common cold" in messages[1].content


@pytest.mark.anyio
async def test_rewrite_rejects_invalid_model_output() -> None:
    chat_model = AsyncMock()
    chat_model.ainvoke.return_value = AIMessage(content="not valid JSON")
    contextualizer = ConversationContextualizer(chat_model)

    with pytest.raises(ConversationContextError, match="invalid output"):
        await contextualizer.rewrite(
            "What are its symptoms?",
            [ContextMessage(role="user", content="What is the common cold?")],
        )
