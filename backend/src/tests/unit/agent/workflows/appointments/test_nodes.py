from datetime import datetime, timezone
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest

from agent.workflows.appointments.nodes import AppointmentNodes
from agent.workflows.appointments.state import AppointmentState as MedicalAgentState
from services.appointment_extractor import (
    AppointmentExtractor,
    ExtractedAppointmentDetails,
)
from services.appointments import (
    Appointment,
    AppointmentConflict,
    AppointmentRequest,
    AppointmentRequestStatus,
    AppointmentService,
)
from services.human_review import (
    HumanReviewCase,
    HumanReviewService,
    ReviewStatus,
    ReviewType,
)


def make_request(
    appointment_request_id: str = "req-123",
    tenant_id: str = "tenant-1",
    conversation_id: str = "conv-456",
    department: str | None = None,
    preferred_date: str | None = None,
    preferred_time: str | None = None,
    contact: str | None = None,
    reason: str | None = None,
    consent_given: bool = False,
    status: AppointmentRequestStatus = "collecting",
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
        status=status,
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
    review_type: ReviewType = "scheduling",
    status: ReviewStatus = "pending",
) -> HumanReviewCase:
    now = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)
    return HumanReviewCase(
        case_id=case_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        run_id=run_id,
        review_type=review_type,
        status=status,
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
        contact="9876543210",
        status="scheduled",
        created_at=datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc),
    )


def make_nodes() -> tuple[
    AppointmentNodes,
    AsyncMock,
    AsyncMock,
    AsyncMock,
]:
    service = AsyncMock(spec=AppointmentService)
    extractor = AsyncMock(spec=AppointmentExtractor)
    human_review = AsyncMock(spec=HumanReviewService)
    nodes = AppointmentNodes(
        appointment_service=service,
        extractor=extractor,
        human_review_service=human_review,
    )
    return nodes, service, extractor, human_review


@pytest.mark.anyio
async def test_ensure_appointment_request_creates_when_missing() -> None:
    nodes, service, _, _ = make_nodes()
    created_request = make_request(appointment_request_id="req-new")
    service.start_request.return_value = created_request

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
    }

    result = await nodes.ensure_appointment_request(state)

    assert result == {"appointment_request_id": "req-new"}
    service.start_request.assert_awaited_once_with(
        tenant_id="tenant-1",
        conversation_id="conv-1",
    )


@pytest.mark.anyio
async def test_ensure_appointment_request_noop_when_existing() -> None:
    nodes, service, _, _ = make_nodes()

    state: MedicalAgentState = {
        "appointment_request_id": "req-already-exists",
    }

    result = await nodes.ensure_appointment_request(state)

    assert result == {}
    service.start_request.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("missing_key", ["tenant_id", "conversation_id"])
async def test_ensure_appointment_request_validates_required_state(
    missing_key: str,
) -> None:
    nodes, _, _, _ = make_nodes()

    state_data: dict[str, str] = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-1",
    }
    state_data.pop(missing_key)
    state = cast(MedicalAgentState, state_data)

    with pytest.raises(ValueError, match=f"Graph state is missing {missing_key}"):
        await nodes.ensure_appointment_request(state)


@pytest.mark.anyio
async def test_collect_appointment_details_updates_when_extracted() -> None:
    nodes, service, extractor, _ = make_nodes()
    existing_request = make_request(appointment_request_id="req-123")
    service.get_request.return_value = existing_request

    extracted_details = ExtractedAppointmentDetails(
        department="Cardiology",
        preferred_date="2026-11-20",
        preferred_time="morning",
        contact=None,
        reason="Heart checkup",
        consent_given=None,
    )
    extractor.extract.return_value = extracted_details

    updated_request = make_request(
        appointment_request_id="req-123",
        department="Cardiology",
        preferred_date="2026-11-20",
        preferred_time="morning",
        reason="Heart checkup",
    )
    service.update_details.return_value = updated_request
    service.missing_fields.return_value = ["contact", "consent_given"]

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "appointment_request_id": "req-123",
        "query": "I need Cardiology on 2026-11-20 in the morning for checkup.",
        "previous_assistant_message": "Which department do you need?",
    }

    result = await nodes.collect_appointment_details(state)

    service.get_request.assert_awaited_once_with(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
    )
    extractor.extract.assert_awaited_once_with(
        patient_message="I need Cardiology on 2026-11-20 in the morning for checkup.",
        current_details=nodes._request_details(existing_request),
        previous_assistant_message="Which department do you need?",
    )
    service.update_details.assert_awaited_once_with(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
        department="Cardiology",
        preferred_date="2026-11-20",
        preferred_time="morning",
        contact=None,
        reason="Heart checkup",
        consent_given=None,
    )
    service.missing_fields.assert_called_once_with(updated_request)
    assert result == {"missing_appointment_fields": ["contact", "consent_given"]}


@pytest.mark.anyio
async def test_collect_appointment_details_skips_update_when_none_extracted() -> None:
    nodes, service, extractor, _ = make_nodes()
    existing_request = make_request(appointment_request_id="req-123")
    service.get_request.return_value = existing_request

    extractor.extract.return_value = ExtractedAppointmentDetails()
    service.missing_fields.return_value = [
        "department",
        "preferred_date",
        "contact",
        "consent_given",
    ]

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "appointment_request_id": "req-123",
        "query": "Hello there",
    }

    result = await nodes.collect_appointment_details(state)

    service.update_details.assert_not_awaited()
    service.missing_fields.assert_called_once_with(existing_request)
    assert result == {
        "missing_appointment_fields": [
            "department",
            "preferred_date",
            "contact",
            "consent_given",
        ]
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "missing_field",
    ["tenant_id", "appointment_request_id", "query"],
)
async def test_collect_appointment_details_validates_required_state(
    missing_field: str,
) -> None:
    nodes, _, _, _ = make_nodes()

    state_data: dict[str, str] = {
        "tenant_id": "tenant-1",
        "appointment_request_id": "req-123",
        "query": "Cardiology",
    }
    state_data.pop(missing_field)
    state = cast(MedicalAgentState, state_data)

    with pytest.raises(ValueError, match=f"Graph state is missing {missing_field}"):
        await nodes.collect_appointment_details(state)


def test_choose_next_appointment_step() -> None:
    nodes, _, _, _ = make_nodes()

    step_with_missing = nodes.choose_next_appointment_step(
        {"missing_appointment_fields": ["department", "contact"]}
    )
    assert step_with_missing == "ask_next_appointment_question"

    step_complete = nodes.choose_next_appointment_step(
        {"missing_appointment_fields": []}
    )
    assert step_complete == "request_appointment_review"

    with pytest.raises(
        ValueError,
        match="missing missing_appointment_fields",
    ):
        nodes.choose_next_appointment_step({})


@pytest.mark.anyio
@pytest.mark.parametrize(
    "field,expected_question",
    [
        (
            "department",
            "Which hospital department or type of care do you need?",
        ),
        (
            "preferred_date",
            "What date would you prefer for the appointment?",
        ),
        (
            "contact",
            "What phone number or email should the hospital use to contact you?",
        ),
        (
            "consent_given",
            "May I send these appointment details to hospital staff for review?",
        ),
    ],
)
async def test_ask_next_appointment_question(
    field: str,
    expected_question: str,
) -> None:
    nodes, _, _, _ = make_nodes()

    state: MedicalAgentState = {"missing_appointment_fields": [field, "other_field"]}

    result = await nodes.ask_next_appointment_question(state)

    assert result == {
        "response": expected_question,
        "previous_assistant_message": expected_question,
    }


@pytest.mark.anyio
async def test_ask_next_appointment_question_when_no_missing_fields() -> None:
    nodes, _, _, _ = make_nodes()

    state: MedicalAgentState = {"missing_appointment_fields": []}

    result = await nodes.ask_next_appointment_question(state)

    assert result == {}


@pytest.mark.anyio
async def test_ask_next_appointment_question_requires_routing_state() -> None:
    nodes, _, _, _ = make_nodes()

    with pytest.raises(
        ValueError,
        match="missing missing_appointment_fields",
    ):
        await nodes.ask_next_appointment_question({})


@pytest.mark.anyio
async def test_ask_next_appointment_question_unknown_field_raises_error() -> None:
    nodes, _, _, _ = make_nodes()

    state: MedicalAgentState = {"missing_appointment_fields": ["unsupported_field"]}

    with pytest.raises(ValueError, match="Unknown field unsupported_field"):
        await nodes.ask_next_appointment_question(state)


def test_request_details_mapping() -> None:
    nodes, _, _, _ = make_nodes()
    req = make_request(
        department="Neurology",
        preferred_date="2026-11-25",
        preferred_time="03:00 PM",
        contact="+1-555-9999",
        reason="Migraine",
        consent_given=True,
    )

    details = nodes._request_details(req)

    assert details == {
        "department": "Neurology",
        "preferred_date": "2026-11-25",
        "preferred_time": "03:00 PM",
        "contact": "+1-555-9999",
        "reason": "Migraine",
        "consent_given": True,
    }


@pytest.mark.anyio
async def test_request_review_from_collecting_creates_case_and_interrupts() -> None:
    nodes, service, _, human_review = make_nodes()
    existing_request = make_request(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
        conversation_id="conv-456",
        status="collecting",
    )
    service.get_request.return_value = existing_request
    created_case = make_review_case(
        case_id="case-999",
        tenant_id="tenant-1",
        conversation_id="conv-456",
        run_id="run-789",
    )
    human_review.request_review.return_value = created_case
    submitted_request = make_request(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
        conversation_id="conv-456",
        status="awaiting_review",
        review_case_id="case-999",
    )
    service.submit_for_review.return_value = submitted_request

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "appointment_request_id": "req-123",
    }

    with patch(
        "agent.workflows.appointments.nodes.interrupt"
    ) as mock_interrupt:
        mock_interrupt.return_value = {"decision": "approved"}
        result = await nodes.request_appointment_review(state)

    human_review.request_review.assert_awaited_once_with(
        tenant_id="tenant-1",
        conversation_id="conv-456",
        run_id="run-789",
        query="Review the completed appointment request req-123",
        review_type="scheduling",
    )
    service.submit_for_review.assert_awaited_once_with(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
        review_case_id="case-999",
    )
    mock_interrupt.assert_called_once_with(
        {
            "type": "appointment_review",
            "case_id": "case-999",
            "appointment_request_id": "req-123",
            "message": (
                "Your appointment request has been sent to hospital staff for review."
            ),
        }
    )
    assert result == {
        "review_case_id": "case-999",
        "review_decision": "approved",
    }


@pytest.mark.anyio
async def test_request_review_from_awaiting_review_bypasses_creation() -> None:
    nodes, service, _, human_review = make_nodes()
    existing_request = make_request(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
        conversation_id="conv-456",
        status="awaiting_review",
        review_case_id="case-existing",
    )
    service.get_request.return_value = existing_request

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "appointment_request_id": "req-123",
    }

    with patch(
        "agent.workflows.appointments.nodes.interrupt"
    ) as mock_interrupt:
        mock_interrupt.return_value = {"decision": "rejected"}
        result = await nodes.request_appointment_review(state)

    human_review.request_review.assert_not_awaited()
    service.submit_for_review.assert_not_awaited()
    mock_interrupt.assert_called_once_with(
        {
            "type": "appointment_review",
            "case_id": "case-existing",
            "appointment_request_id": "req-123",
            "message": (
                "Your appointment request has been sent to hospital staff for review."
            ),
        }
    )
    assert result == {
        "review_case_id": "case-existing",
        "review_decision": "rejected",
    }


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["approved", "booked", "rejected"])
async def test_request_appointment_review_conflict_when_invalid_status(
    status: AppointmentRequestStatus,
) -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(status=status)

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "appointment_request_id": "req-123",
    }

    with pytest.raises(
        AppointmentConflict,
        match=f"Appointment request cannot be reviewed from {status}",
    ):
        await nodes.request_appointment_review(state)


@pytest.mark.anyio
async def test_request_appointment_review_conflict_when_no_case_id() -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(
        status="awaiting_review",
        review_case_id=None,
    )

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "appointment_request_id": "req-123",
    }

    with pytest.raises(
        AppointmentConflict,
        match="Appointment request has no review case",
    ):
        await nodes.request_appointment_review(state)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "missing_key",
    ["tenant_id", "conversation_id", "run_id", "appointment_request_id"],
)
async def test_request_appointment_review_validates_required_state(
    missing_key: str,
) -> None:
    nodes, _, _, _ = make_nodes()

    state_data: dict[str, str] = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "appointment_request_id": "req-123",
    }
    state_data.pop(missing_key)
    state = cast(MedicalAgentState, state_data)

    with pytest.raises(ValueError, match=f"Graph state is missing {missing_key}"):
        await nodes.request_appointment_review(state)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_resume,expected_error",
    [
        ("approved", "Human-review resume value must be an object"),
        ({"decision": "unknown"}, "Human-review decision must be approved or rejected"),
        ({}, "Human-review decision must be approved or rejected"),
    ],
)
async def test_request_appointment_review_validates_resume_value(
    invalid_resume: object,
    expected_error: str,
) -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(
        status="awaiting_review",
        review_case_id="case-123",
    )

    state: MedicalAgentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-456",
        "run_id": "run-789",
        "appointment_request_id": "req-123",
    }

    with patch(
        "agent.workflows.appointments.nodes.interrupt",
        return_value=invalid_resume,
    ):
        with pytest.raises(ValueError, match=expected_error):
            await nodes.request_appointment_review(state)


def test_approved_review_routes_to_booking() -> None:
    nodes, _, _, _ = make_nodes()

    result = nodes.choose_after_appointment_review({"review_decision": "approved"})

    assert result == "book_appointment"


def test_rejected_review_routes_to_rejection() -> None:
    nodes, _, _, _ = make_nodes()

    result = nodes.choose_after_appointment_review({"review_decision": "rejected"})

    assert result == "appointment_rejected"


def test_review_route_requires_decision() -> None:
    nodes, _, _, _ = make_nodes()

    with pytest.raises(
        ValueError,
        match="valid review_decision",
    ):
        nodes.choose_after_appointment_review({})


@pytest.mark.anyio
async def test_book_approved_appointment() -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(
        status="awaiting_review",
        review_case_id="case-123",
    )
    service.approve.return_value = make_request(
        status="approved",
        review_case_id="case-123",
    )
    service.book.return_value = make_appointment()

    result = await nodes.book_appointment(
        {
            "tenant_id": "tenant-1",
            "appointment_request_id": "req-123",
            "review_decision": "approved",
        }
    )

    service.approve.assert_awaited_once_with(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
    )
    service.book.assert_awaited_once_with(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
    )
    assert result["appointment_id"] == "appointment-123"
    assert "approved" in str(result["response"])


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["approved", "booked"])
async def test_booking_retry_does_not_approve_again(
    status: AppointmentRequestStatus,
) -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(
        status=status,
        review_case_id="case-123",
    )
    service.book.return_value = make_appointment()

    result = await nodes.book_appointment(
        {
            "tenant_id": "tenant-1",
            "appointment_request_id": "req-123",
            "review_decision": "approved",
        }
    )

    service.approve.assert_not_awaited()
    service.book.assert_awaited_once()
    assert result["appointment_id"] == "appointment-123"


@pytest.mark.anyio
async def test_booking_requires_approved_decision() -> None:
    nodes, service, _, _ = make_nodes()

    with pytest.raises(ValueError, match="approved review decision"):
        await nodes.book_appointment(
            {
                "tenant_id": "tenant-1",
                "appointment_request_id": "req-123",
                "review_decision": "rejected",
            }
        )

    service.get_request.assert_not_awaited()
    service.book.assert_not_awaited()


@pytest.mark.anyio
async def test_booking_rejects_invalid_request_status() -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(status="collecting")

    with pytest.raises(
        AppointmentConflict,
        match="cannot be booked from collecting",
    ):
        await nodes.book_appointment(
            {
                "tenant_id": "tenant-1",
                "appointment_request_id": "req-123",
                "review_decision": "approved",
            }
        )

    service.book.assert_not_awaited()


@pytest.mark.anyio
async def test_rejected_request_is_recorded() -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(
        status="awaiting_review",
        review_case_id="case-123",
    )
    service.reject.return_value = make_request(
        status="rejected",
        review_case_id="case-123",
    )

    result = await nodes.appointment_rejected(
        {
            "tenant_id": "tenant-1",
            "appointment_request_id": "req-123",
            "review_decision": "rejected",
        }
    )

    service.reject.assert_awaited_once_with(
        appointment_request_id="req-123",
        tenant_id="tenant-1",
    )
    service.book.assert_not_awaited()
    assert "could not approve" in str(result["response"])


@pytest.mark.anyio
async def test_rejection_retry_does_not_reject_again() -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(
        status="rejected",
        review_case_id="case-123",
    )

    result = await nodes.appointment_rejected(
        {
            "tenant_id": "tenant-1",
            "appointment_request_id": "req-123",
            "review_decision": "rejected",
        }
    )

    service.reject.assert_not_awaited()
    assert "could not approve" in str(result["response"])


@pytest.mark.anyio
async def test_rejection_requires_rejected_decision() -> None:
    nodes, service, _, _ = make_nodes()

    with pytest.raises(ValueError, match="rejected review decision"):
        await nodes.appointment_rejected(
            {
                "tenant_id": "tenant-1",
                "appointment_request_id": "req-123",
                "review_decision": "approved",
            }
        )

    service.get_request.assert_not_awaited()
    service.reject.assert_not_awaited()


@pytest.mark.anyio
async def test_rejection_rejects_invalid_request_status() -> None:
    nodes, service, _, _ = make_nodes()
    service.get_request.return_value = make_request(status="approved")

    with pytest.raises(
        AppointmentConflict,
        match="cannot be rejected from approved",
    ):
        await nodes.appointment_rejected(
            {
                "tenant_id": "tenant-1",
                "appointment_request_id": "req-123",
                "review_decision": "rejected",
            }
        )

    service.reject.assert_not_awaited()
