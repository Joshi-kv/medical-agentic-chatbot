from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.workflows.medical_agent.graph import build_medical_agent_graph
from agent.workflows.medical_agent.nodes import (
    HUMAN_REVIEW_MESSAGE,
    ROUTING_FAILURE_MESSAGE,
    MedicalAgentNodes,
)
from guardrails.emergency import EmergencyGuardrail, EmergencyRule
from guardrails.models import AgentRoute, RouteDecision
from services.answering import (
    SAFETY_NOTICE,
    AnswerCompleted,
    AnswerDelta,
    AnswerProgress,
    AnswerResult,
)
from services.conversation_context import ContextMessage, ConversationContextError
from services.intent_router import IntentRouterError
from services.medical_agent import MedicalAgent


def make_agent(
    contextualizer: AsyncMock | None = None,
) -> tuple[MedicalAgent, AsyncMock, AsyncMock, AsyncMock]:
    router = AsyncMock()
    answering = AsyncMock()
    appointments = AsyncMock()
    contextualizer = contextualizer or AsyncMock()
    appointments.has_active_request.return_value = False
    nodes = MedicalAgentNodes(
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
        contextualizer=contextualizer,
    )
    agent = MedicalAgent(graph=build_medical_agent_graph(nodes=nodes))
    return agent, router, answering, appointments


@pytest.mark.anyio
async def test_follow_up_uses_contextualized_query_for_routing_and_answering() -> None:
    contextualizer = AsyncMock()
    contextualizer.rewrite.return_value = (
        "What are the symptoms of the common cold?"
    )
    agent, router, answering, _ = make_agent(contextualizer)
    router.classify.return_value = RouteDecision(
        route=AgentRoute.EDUCATION,
        confidence=0.95,
    )
    answering.answer.return_value = AnswerResult(
        answer="Common cold symptoms include congestion [C1].",
        citations=[],
        sufficient_evidence=True,
        safety_notice=SAFETY_NOTICE,
    )
    history = [
        ContextMessage(role="user", content="What is the common cold?"),
        ContextMessage(role="assistant", content="It is a viral infection."),
    ]

    await agent.answer(
        query="What are its symptoms?",
        tenant_id="tenant-1",
        conversation_history=history,
    )

    contextualizer.rewrite.assert_awaited_once_with(
        query="What are its symptoms?",
        history=history,
    )
    router.classify.assert_awaited_once_with(
        "What are the symptoms of the common cold?"
    )
    answering.answer.assert_awaited_once_with(
        query="What are the symptoms of the common cold?",
        tenant_id="tenant-1",
        document_id=None,
    )


@pytest.mark.anyio
async def test_contextualizer_failure_falls_back_to_original_query() -> None:
    contextualizer = AsyncMock()
    contextualizer.rewrite.side_effect = ConversationContextError("unavailable")
    agent, router, answering, _ = make_agent(contextualizer)
    router.classify.return_value = RouteDecision(
        route=AgentRoute.CLARIFICATION,
        confidence=0.9,
        follow_up_question="Which condition do you mean?",
    )

    result = await agent.answer(
        query="What are its symptoms?",
        tenant_id="tenant-1",
        conversation_history=[
            ContextMessage(role="user", content="What is the common cold?")
        ],
    )

    router.classify.assert_awaited_once_with("What are its symptoms?")
    answering.answer.assert_not_awaited()
    assert result.answer == "Which condition do you mean?"


@pytest.mark.anyio
async def test_contextualized_emergency_stops_before_router() -> None:
    contextualizer = AsyncMock()
    contextualizer.rewrite.return_value = "approved emergency test phrase"
    agent, router, answering, _ = make_agent(contextualizer)

    result = await agent.answer(
        query="It is getting worse",
        tenant_id="tenant-1",
        conversation_history=[
            ContextMessage(role="user", content="I have chest pain")
        ],
    )

    assert result.route == "emergency"
    router.classify.assert_not_awaited()
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_active_appointment_does_not_use_general_contextualizer() -> None:
    contextualizer = AsyncMock()
    agent, router, _, appointments = make_agent(contextualizer)
    appointments.has_active_request.return_value = True
    appointments.run.return_value = AnswerResult(
        answer="What date would you prefer?",
        citations=[],
        sufficient_evidence=False,
        safety_notice=SAFETY_NOTICE,
        route="appointment_request",
    )

    await agent.answer(
        query="Tomorrow morning",
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        run_id="run-1",
        conversation_history=[
            ContextMessage(role="assistant", content="What date do you prefer?")
        ],
    )

    contextualizer.rewrite.assert_not_awaited()
    router.classify.assert_not_awaited()


@pytest.mark.anyio
async def test_stream_uses_contextualized_query() -> None:
    contextualizer = AsyncMock()
    contextualizer.rewrite.return_value = (
        "What are the symptoms of the common cold?"
    )
    agent, router, answering, _ = make_agent(contextualizer)
    router.classify.return_value = RouteDecision(
        route=AgentRoute.EDUCATION,
        confidence=0.95,
    )
    result = AnswerResult(
        answer="Symptoms include congestion [C1].",
        citations=[],
        sufficient_evidence=True,
        safety_notice=SAFETY_NOTICE,
    )

    async def answer_stream(**kwargs: object):
        yield AnswerCompleted(result=result)

    answering.stream_answer = MagicMock(side_effect=answer_stream)
    history = [
        ContextMessage(role="user", content="What is the common cold?"),
        ContextMessage(role="assistant", content="It is a viral infection."),
    ]

    events = [
        event
        async for event in agent.stream_answer(
            query="What are its symptoms?",
            tenant_id="tenant-1",
            conversation_history=history,
        )
    ]

    assert any(
        isinstance(event, AnswerProgress) and event.stage == "context"
        for event in events
    )
    answering.stream_answer.assert_called_once_with(
        query="What are the symptoms of the common cold?",
        tenant_id="tenant-1",
        document_id=None,
    )


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
