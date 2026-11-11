from unittest.mock import AsyncMock

import pytest

from guardrails.emergency import EmergencyGuardrail, EmergencyRule
from guardrails.models import AgentRoute, RouteDecision
from services.answering import (
    SAFETY_NOTICE,
    AnswerCompleted,
    AnswerDelta,
    AnswerResult,
)
from services.intent_router import IntentRouterError
from services.medical_agent import (
    APPOINTMENT_MESSAGE,
    HUMAN_REVIEW_MESSAGE,
    ROUTING_FAILURE_MESSAGE,
    MedicalAgent,
)


def make_agent() -> tuple[MedicalAgent, AsyncMock, AsyncMock]:
    router = AsyncMock()
    answering = AsyncMock()
    agent = MedicalAgent(
        emergency_guardrail=EmergencyGuardrail(
            rules=(
                EmergencyRule(
                    name="test_emergency",
                    phrases=("approved emergency test phrase",),
                ),
            )
        ),
        intent_router=router,
        answering_service=answering,
    )
    return agent, router, answering


@pytest.mark.anyio
async def test_emergency_stops_before_router_and_rag() -> None:
    agent, router, answering = make_agent()

    result = await agent.answer(
        query="This contains an approved emergency test phrase.",
        tenant_id="tenant-1",
    )

    assert result.route == "emergency"
    assert "emergency" in result.answer.lower()
    router.classify.assert_not_awaited()
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_education_uses_existing_answering_service() -> None:
    agent, router, answering = make_agent()
    router.classify.return_value = RouteDecision(
        route=AgentRoute.EDUCATION,
        confidence=0.95,
    )
    answering.answer.return_value = AnswerResult(
        answer="Evidence-based answer [C1]",
        citations=[],
        sufficient_evidence=True,
        safety_notice=SAFETY_NOTICE,
    )

    result = await agent.answer(
        query="What is hypertension?",
        tenant_id="tenant-1",
    )

    assert result.answer == "Evidence-based answer [C1]"
    assert result.route == "curated_documents"
    answering.answer.assert_awaited_once()


@pytest.mark.anyio
async def test_appointment_request_does_not_book() -> None:
    agent, router, answering = make_agent()
    router.classify.return_value = RouteDecision(
        route=AgentRoute.APPOINTMENT_REQUEST,
        confidence=0.9,
    )

    result = await agent.answer(
        query="I want an appointment tomorrow.",
        tenant_id="tenant-1",
    )

    assert result.answer == APPOINTMENT_MESSAGE
    assert result.route == AgentRoute.APPOINTMENT_REQUEST
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_human_review_does_not_generate_advice() -> None:
    agent, router, answering = make_agent()
    router.classify.return_value = RouteDecision(
        route=AgentRoute.HUMAN_REVIEW,
        confidence=0.96,
    )

    result = await agent.answer(
        query="Should I stop my prescription?",
        tenant_id="tenant-1",
    )

    assert result.answer == HUMAN_REVIEW_MESSAGE
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_router_failure_fails_closed() -> None:
    agent, router, answering = make_agent()
    router.classify.side_effect = IntentRouterError("provider failed")

    result = await agent.answer(
        query="Help me",
        tenant_id="tenant-1",
    )

    assert result.answer == ROUTING_FAILURE_MESSAGE
    assert result.route == AgentRoute.HUMAN_REVIEW
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_emergency_stream_stops_before_router() -> None:
    agent, router, answering = make_agent()

    events = [
        event
        async for event in agent.stream_answer(
            query="Approved emergency test phrase",
            tenant_id="tenant-1",
        )
    ]

    assert isinstance(events[-2], AnswerDelta)
    assert isinstance(events[-1], AnswerCompleted)
    assert events[-1].result.route == "emergency"
    router.classify.assert_not_awaited()
    answering.stream_answer.assert_not_called()
