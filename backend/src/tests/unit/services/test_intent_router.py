from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from guardrails.models import AgentRoute
from services.intent_router import IntentRouter, IntentRouterError


@pytest.mark.anyio
async def test_classifies_supported_route() -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content='{"route":"education","confidence":0.94,'
        '"follow_up_question":null}'
    )
    router = IntentRouter(chat_model=model)

    result = await router.classify("What is hypertension?")

    assert result.route == AgentRoute.EDUCATION
    assert result.confidence == 0.94
    assert result.follow_up_question is None


@pytest.mark.anyio
async def test_low_confidence_asks_for_clarification() -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content='{"route":"education","confidence":0.4,'
        '"follow_up_question":null}'
    )
    router = IntentRouter(chat_model=model)

    result = await router.classify("Help me")

    assert result.route == AgentRoute.CLARIFICATION
    assert result.follow_up_question is not None


@pytest.mark.anyio
async def test_empty_message_does_not_call_model() -> None:
    model = AsyncMock()
    router = IntentRouter(chat_model=model)

    result = await router.classify("   ")

    assert result.route == AgentRoute.CLARIFICATION
    model.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_invalid_model_response_fails_closed() -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content="not json")
    router = IntentRouter(chat_model=model)

    with pytest.raises(IntentRouterError, match="classification failed"):
        await router.classify("What should I do?")


def test_minimum_confidence_is_validated() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        IntentRouter(chat_model=AsyncMock(), minimum_confidence=1.1)
