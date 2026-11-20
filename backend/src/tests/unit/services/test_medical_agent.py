from unittest.mock import AsyncMock

import pytest

from guardrails.emergency import EmergencyGuardrail, EmergencyRule
from guardrails.models import AgentRoute, RouteDecision
from services.answering import (
    SAFETY_NOTICE,
    AnswerCompleted,
    AnswerDelta,
    AnswerProgress,
    AnswerResult,
)
from services.intent_router import IntentRouterError
from services.medical_agent import (
    HUMAN_REVIEW_MESSAGE,
    ROUTING_FAILURE_MESSAGE,
    MedicalAgent,
)


def make_agent() -> tuple[MedicalAgent, AsyncMock, AsyncMock, AsyncMock]:
    router = AsyncMock()
    answering = AsyncMock()
    appointments = AsyncMock()
    appointments.has_active_request.return_value = False
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
        appointment_workflow=appointments,
    )
    return agent, router, answering, appointments


@pytest.mark.anyio
async def test_emergency_stops_before_router_and_rag() -> None:
    agent, router, answering, _ = make_agent()

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
    agent, router, answering, _ = make_agent()
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
async def test_appointment_request_runs_workflow() -> None:
    agent, router, answering, appointments = make_agent()
    router.classify.return_value = RouteDecision(
        route=AgentRoute.APPOINTMENT_REQUEST,
        confidence=0.9,
    )

    appointments.run.return_value = AnswerResult(
        answer="What date would you prefer?",
        citations=[],
        sufficient_evidence=False,
        safety_notice=SAFETY_NOTICE,
        route="appointment_request",
    )

    result = await agent.answer(
        query="I want an appointment tomorrow.",
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        run_id="run-1",
    )

    assert result.answer == "What date would you prefer?"
    assert result.route == AgentRoute.APPOINTMENT_REQUEST
    appointments.run.assert_awaited_once_with(
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        run_id="run-1",
        query="I want an appointment tomorrow.",
    )
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_human_review_does_not_generate_advice() -> None:
    agent, router, answering, _ = make_agent()
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
    agent, router, answering, _ = make_agent()
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
    agent, router, answering, _ = make_agent()

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


@pytest.mark.anyio
async def test_active_appointment_bypasses_router() -> None:
    agent, router, answering, appointments = make_agent()
    appointments.has_active_request.return_value = True
    appointments.run.return_value = AnswerResult(
        answer="What phone number should the hospital use?",
        citations=[],
        sufficient_evidence=False,
        safety_notice=SAFETY_NOTICE,
        route="appointment_request",
    )

    result = await agent.answer(
        query="Tomorrow morning.",
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        run_id="run-2",
    )

    assert result.answer == "What phone number should the hospital use?"
    router.classify.assert_not_awaited()
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_appointment_stream_returns_graph_answer() -> None:
    agent, router, _, appointments = make_agent()
    router.classify.return_value = RouteDecision(
        route=AgentRoute.APPOINTMENT_REQUEST,
        confidence=0.95,
    )
    appointments.run.return_value = AnswerResult(
        answer="Which department do you need?",
        citations=[],
        sufficient_evidence=False,
        safety_notice=SAFETY_NOTICE,
        route="appointment_request",
    )

    events = [
        event
        async for event in agent.stream_answer(
            query="Book an appointment.",
            tenant_id="tenant-1",
            conversation_id="conversation-1",
            run_id="run-1",
        )
    ]

    assert any(
        isinstance(event, AnswerProgress) and event.stage == "appointment"
        for event in events
    )
    assert isinstance(events[-2], AnswerDelta)
    assert events[-2].text == "Which department do you need?"
    assert isinstance(events[-1], AnswerCompleted)
