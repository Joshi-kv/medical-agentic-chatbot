from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from agent.graph import build_appointment_graph
from agent.nodes.appointments import AppointmentNodes
from agent.state import MedicalAgentState
from services.appointment_extractor import (
    AppointmentExtractor,
    ExtractedAppointmentDetails,
)
from services.appointments import (
    Appointment,
    AppointmentRequest,
    AppointmentService,
)
from services.human_review import HumanReviewCase, HumanReviewService


def make_request(
    appointment_request_id: str = "req-123",
    tenant_id: str = "tenant-1",
    conversation_id: str = "conv-456",
    department: str | None = "Cardiology",
    preferred_date: str | None = "2026-11-20",
    preferred_time: str | None = "morning",
    contact: str | None = "+1234567890",
    reason: str | None = "Routine checkup",
    consent_given: bool = True,
    status: str = "collecting",
    review_case_id: str | None = None,
) -> AppointmentRequest:
    now = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)
    return AppointmentRequest(
        appointment_request_id=appointment_request_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        department=department,
        preferred_date=preferred_date,
        preferred_time=preferred_time,
        contact=contact,
        reason=reason,
        consent_given=consent_given,
        status=status,  # type: ignore[arg-type]
        review_case_id=review_case_id,
        appointment_id=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )


def make_review_case(
    case_id: str = "case-123",
    tenant_id: str = "tenant-1",
    conversation_id: str = "conv-456",
    run_id: str = "run-789",
) -> HumanReviewCase:
    now = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)
    return HumanReviewCase(
        case_id=case_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        run_id=run_id,
        review_type="scheduling",
        status="pending",
        requested_action="review_user_request",
        context={"query": "Review the completed appointment request req-123"},
        claimed_by=None,
        claimed_at=None,
        decision=None,
        resolution=None,
        resolved_by=None,
        resolved_at=None,
        expires_at=now,
        created_at=now,
        updated_at=now,
    )


def make_appointment() -> Appointment:
    return Appointment(
        appointment_id="appointment-123",
        tenant_id="tenant-1",
        appointment_request_id="req-123",
        department="Cardiology",
        scheduled_date="2026-11-20",
        scheduled_time="morning",
        contact="+1234567890",
        status="scheduled",
        created_at=datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc),
    )


def test_build_appointment_graph_compiles() -> None:
    nodes = MagicMock(spec=AppointmentNodes)
    checkpointer = MemorySaver()

    graph = build_appointment_graph(nodes=nodes, checkpointer=checkpointer)

    assert isinstance(graph, CompiledStateGraph)
    assert graph.name == "appointment_workflow"
    assert graph.checkpointer is checkpointer

    expected_nodes = {
        "__start__",
        "ensure_appointment_request",
        "collect_appointment_details",
        "ask_next_appointment_question",
        "request_appointment_review",
        "book_appointment",
        "appointment_rejected",
        "__end__",
    }
    actual_nodes = set(graph.get_graph().nodes.keys())
    assert expected_nodes.issubset(actual_nodes)


@pytest.mark.anyio
async def test_graph_routes_to_ask_next_question_when_missing_fields() -> None:
    nodes = MagicMock(spec=AppointmentNodes)
    nodes.ensure_appointment_request = AsyncMock(
        return_value={"appointment_request_id": "req-123"}
    )
    nodes.collect_appointment_details = AsyncMock(
        return_value={"missing_appointment_fields": ["department"]}
    )
    nodes.choose_next_appointment_step = MagicMock(
        return_value="ask_next_appointment_question"
    )
    nodes.ask_next_appointment_question = AsyncMock(
        return_value={
            "response": "Which hospital department or type of care do you need?",
            "previous_assistant_message": (
                "Which hospital department or type of care do you need?"
            ),
        }
    )

    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-1"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "query": "I want to schedule an appointment",
    }

    result = await graph.ainvoke(state, config=config)

    assert result["appointment_request_id"] == "req-123"
    assert result["missing_appointment_fields"] == ["department"]
    assert (
        result["response"] == "Which hospital department or type of care do you need?"
    )

    nodes.ensure_appointment_request.assert_awaited_once()
    nodes.collect_appointment_details.assert_awaited_once()
    nodes.ask_next_appointment_question.assert_awaited_once()
    nodes.request_appointment_review.assert_not_awaited()
    nodes.book_appointment.assert_not_awaited()
    nodes.appointment_rejected.assert_not_awaited()


@pytest.mark.anyio
async def test_graph_routes_to_review_and_interrupts() -> None:
    async def mock_review(
        current_state: MedicalAgentState,
    ) -> dict[str, object]:
        resume_val = interrupt(
            {
                "type": "appointment_review",
                "case_id": "case-123",
                "appointment_request_id": "req-123",
                "message": "Appointment request sent for review.",
            }
        )
        return {
            "review_case_id": "case-123",
            "review_decision": resume_val["decision"],
        }

    nodes = MagicMock(spec=AppointmentNodes)
    nodes.ensure_appointment_request = AsyncMock(
        return_value={"appointment_request_id": "req-123"}
    )
    nodes.collect_appointment_details = AsyncMock(
        return_value={"missing_appointment_fields": []}
    )
    nodes.choose_next_appointment_step = MagicMock(
        return_value="request_appointment_review"
    )
    nodes.request_appointment_review = mock_review
    nodes.choose_after_appointment_review = MagicMock(return_value="book_appointment")
    nodes.book_appointment = AsyncMock(
        return_value={
            "appointment_id": "appointment-123",
            "response": "Your appointment request has been approved and scheduled.",
        }
    )

    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-2"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "query": "Details complete",
    }

    # Initial invoke triggers interrupt in request_appointment_review
    await graph.ainvoke(state, config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("request_appointment_review",)
    assert len(snapshot.tasks) == 1
    interrupt_task = snapshot.tasks[0]
    assert len(interrupt_task.interrupts) == 1
    assert interrupt_task.interrupts[0].value == {
        "type": "appointment_review",
        "case_id": "case-123",
        "appointment_request_id": "req-123",
        "message": "Appointment request sent for review.",
    }


@pytest.mark.anyio
async def test_graph_routes_approved_review_to_booking() -> None:
    async def mock_review(
        current_state: MedicalAgentState,
    ) -> dict[str, object]:
        resume_val = interrupt(
            {
                "type": "appointment_review",
                "case_id": "case-123",
                "appointment_request_id": "req-123",
            }
        )
        return {
            "review_case_id": "case-123",
            "review_decision": resume_val["decision"],
        }

    nodes = MagicMock(spec=AppointmentNodes)
    nodes.ensure_appointment_request = AsyncMock(
        return_value={"appointment_request_id": "req-123"}
    )
    nodes.collect_appointment_details = AsyncMock(
        return_value={"missing_appointment_fields": []}
    )
    nodes.choose_next_appointment_step = MagicMock(
        return_value="request_appointment_review"
    )
    nodes.request_appointment_review = mock_review

    def mock_route_decision(current_state: MedicalAgentState) -> str:
        if current_state.get("review_decision") == "approved":
            return "book_appointment"
        return "appointment_rejected"

    nodes.choose_after_appointment_review = mock_route_decision
    nodes.book_appointment = AsyncMock(
        return_value={
            "appointment_id": "appointment-123",
            "response": "Your appointment request has been approved and scheduled.",
        }
    )
    nodes.appointment_rejected = AsyncMock()

    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-3"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "query": "Booking ready",
    }

    # Step 1: pause on interrupt
    await graph.ainvoke(state, config=config)

    # Step 2: resume with approval decision
    resumed_result = await graph.ainvoke(
        Command(resume={"decision": "approved"}),
        config=config,
    )

    assert resumed_result["review_decision"] == "approved"
    assert resumed_result["appointment_id"] == "appointment-123"
    assert "approved and scheduled" in str(resumed_result["response"])

    nodes.book_appointment.assert_awaited_once()
    nodes.appointment_rejected.assert_not_awaited()

    final_snapshot = await graph.aget_state(config)
    assert final_snapshot.next == ()


@pytest.mark.anyio
async def test_graph_routes_rejected_review_to_rejection() -> None:
    async def mock_review(
        current_state: MedicalAgentState,
    ) -> dict[str, object]:
        resume_val = interrupt(
            {
                "type": "appointment_review",
                "case_id": "case-123",
                "appointment_request_id": "req-123",
            }
        )
        return {
            "review_case_id": "case-123",
            "review_decision": resume_val["decision"],
        }

    nodes = MagicMock(spec=AppointmentNodes)
    nodes.ensure_appointment_request = AsyncMock(
        return_value={"appointment_request_id": "req-123"}
    )
    nodes.collect_appointment_details = AsyncMock(
        return_value={"missing_appointment_fields": []}
    )
    nodes.choose_next_appointment_step = MagicMock(
        return_value="request_appointment_review"
    )
    nodes.request_appointment_review = mock_review

    def mock_route_decision(current_state: MedicalAgentState) -> str:
        if current_state.get("review_decision") == "approved":
            return "book_appointment"
        return "appointment_rejected"

    nodes.choose_after_appointment_review = mock_route_decision
    nodes.book_appointment = AsyncMock()
    nodes.appointment_rejected = AsyncMock(
        return_value={
            "response": "Hospital staff could not approve this appointment request.",
        }
    )

    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-4"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "query": "Booking ready",
    }

    # Step 1: pause on interrupt
    await graph.ainvoke(state, config=config)

    # Step 2: resume with rejection decision
    resumed_result = await graph.ainvoke(
        Command(resume={"decision": "rejected"}),
        config=config,
    )

    assert resumed_result["review_decision"] == "rejected"
    assert "could not approve" in str(resumed_result["response"])

    nodes.appointment_rejected.assert_awaited_once()
    nodes.book_appointment.assert_not_awaited()

    final_snapshot = await graph.aget_state(config)
    assert final_snapshot.next == ()


@pytest.mark.anyio
async def test_e2e_appointment_workflow_missing_field_to_question() -> None:
    service = AsyncMock(spec=AppointmentService)
    extractor = AsyncMock(spec=AppointmentExtractor)
    human_review = AsyncMock(spec=HumanReviewService)

    req = make_request(
        appointment_request_id="req-123",
        status="collecting",
    )
    service.start_request.return_value = req
    service.get_request.return_value = req
    extractor.extract.return_value = ExtractedAppointmentDetails(
        department="Cardiology"
    )
    service.update_details.return_value = req
    service.missing_fields.return_value = ["preferred_date", "contact"]

    nodes = AppointmentNodes(
        appointment_service=service,
        extractor=extractor,
        human_review_service=human_review,
    )
    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-e2e-1"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "query": "I need to see a cardiologist",
    }

    result = await graph.ainvoke(state, config=config)

    assert result["appointment_request_id"] == "req-123"
    assert result["missing_appointment_fields"] == ["preferred_date", "contact"]
    assert result["response"] == "What date would you prefer for the appointment?"


@pytest.mark.anyio
async def test_e2e_appointment_workflow_complete_to_booking() -> None:
    service = AsyncMock(spec=AppointmentService)
    extractor = AsyncMock(spec=AppointmentExtractor)
    human_review = AsyncMock(spec=HumanReviewService)

    initial_req = make_request(status="collecting")
    service.start_request.return_value = initial_req
    service.get_request.return_value = initial_req
    extractor.extract.return_value = ExtractedAppointmentDetails()
    service.missing_fields.return_value = []

    review_case = make_review_case(case_id="case-999")
    human_review.request_review.return_value = review_case

    awaiting_req = make_request(
        status="awaiting_review",
        review_case_id="case-999",
    )
    service.submit_for_review.return_value = awaiting_req

    nodes = AppointmentNodes(
        appointment_service=service,
        extractor=extractor,
        human_review_service=human_review,
    )
    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-e2e-2"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "query": "All details confirmed",
    }

    # Step 1: Initial invoke pauses at interrupt
    await graph.ainvoke(state, config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("request_appointment_review",)
    assert snapshot.tasks[0].interrupts[0].value["case_id"] == "case-999"

    # Step 2: Staff approves via resume
    service.get_request.return_value = awaiting_req
    service.approve.return_value = make_request(
        status="approved",
        review_case_id="case-999",
    )
    service.book.return_value = make_appointment()

    resumed_result = await graph.ainvoke(
        Command(resume={"decision": "approved"}),
        config=config,
    )

    assert resumed_result["review_decision"] == "approved"
    assert resumed_result["appointment_id"] == "appointment-123"
    assert "approved and scheduled" in str(resumed_result["response"])


@pytest.mark.anyio
async def test_e2e_appointment_workflow_complete_to_rejection() -> None:
    service = AsyncMock(spec=AppointmentService)
    extractor = AsyncMock(spec=AppointmentExtractor)
    human_review = AsyncMock(spec=HumanReviewService)

    initial_req = make_request(status="collecting")
    service.start_request.return_value = initial_req
    service.get_request.return_value = initial_req
    extractor.extract.return_value = ExtractedAppointmentDetails()
    service.missing_fields.return_value = []

    review_case = make_review_case(case_id="case-999")
    human_review.request_review.return_value = review_case

    awaiting_req = make_request(
        status="awaiting_review",
        review_case_id="case-999",
    )
    service.submit_for_review.return_value = awaiting_req

    nodes = AppointmentNodes(
        appointment_service=service,
        extractor=extractor,
        human_review_service=human_review,
    )
    checkpointer = MemorySaver()
    graph = build_appointment_graph(nodes, checkpointer)
    config = {"configurable": {"thread_id": "thread-e2e-3"}}

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "query": "All details confirmed",
    }

    # Step 1: Initial invoke pauses at interrupt
    await graph.ainvoke(state, config=config)

    # Step 2: Staff rejects via resume
    service.get_request.return_value = awaiting_req
    service.reject.return_value = make_request(
        status="rejected",
        review_case_id="case-999",
    )

    resumed_result = await graph.ainvoke(
        Command(resume={"decision": "rejected"}),
        config=config,
    )

    assert resumed_result["review_decision"] == "rejected"
    assert "could not approve" in str(resumed_result["response"])
