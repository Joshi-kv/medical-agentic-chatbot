from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from services.appointments import (
    Appointment,
    AppointmentConflict,
    AppointmentError,
    AppointmentNotFound,
    AppointmentRequest,
    AppointmentRequestStatus,
    AppointmentService,
    InvalidAppointmentInput,
)

REQUEST_ID = "11111111-1111-4111-8111-111111111111"
CONVERSATION_ID = "22222222-2222-4222-8222-222222222222"
APPOINTMENT_ID = "33333333-3333-4333-8333-333333333333"
REVIEW_CASE_ID = "44444444-4444-4444-8444-444444444444"


def make_request(
    status: AppointmentRequestStatus = "collecting",
    department: str | None = "Cardiology",
    preferred_date: str | None = "2026-11-20",
    preferred_time: str | None = "10:00 AM",
    contact: str | None = "+1-555-0100",
    reason: str | None = "Chest tightness follow-up",
    consent_given: bool = True,
    review_case_id: str | None = None,
    appointment_id: str | None = None,
) -> AppointmentRequest:
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    return AppointmentRequest(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        conversation_id=CONVERSATION_ID,
        department=department,
        preferred_date=preferred_date,
        preferred_time=preferred_time,
        contact=contact,
        reason=reason,
        consent_given=consent_given,
        status=status,
        review_case_id=review_case_id,
        appointment_id=appointment_id,
        error_message=None,
        created_at=now,
        updated_at=now,
    )


def make_appointment() -> Appointment:
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    return Appointment(
        appointment_id=APPOINTMENT_ID,
        tenant_id="tenant-1",
        appointment_request_id=REQUEST_ID,
        department="Cardiology",
        scheduled_date="2026-11-20",
        scheduled_time="10:00 AM",
        contact="+1-555-0100",
        status="scheduled",
        created_at=now,
    )


@pytest.mark.anyio
async def test_start_request_creates_and_returns_request() -> None:
    store = AsyncMock()
    initial_request = make_request(
        status="collecting",
        department=None,
        preferred_date=None,
        preferred_time=None,
        contact=None,
        reason=None,
        consent_given=False,
    )
    store.get_or_create_request.return_value = initial_request
    service = AppointmentService(store=store)

    result = await service.start_request(
        tenant_id="tenant-1",
        conversation_id=CONVERSATION_ID,
    )

    assert result == initial_request
    store.get_or_create_request.assert_awaited_once()
    saved = store.get_or_create_request.await_args.args[0]
    assert isinstance(saved, AppointmentRequest)
    assert saved.tenant_id == "tenant-1"
    assert saved.conversation_id == CONVERSATION_ID
    assert saved.status == "collecting"
    assert saved.consent_given is False


@pytest.mark.anyio
@pytest.mark.parametrize("empty_val", ["", "   "])
async def test_start_request_validates_required_fields(empty_val: str) -> None:
    store = AsyncMock()
    service = AppointmentService(store=store)

    with pytest.raises(InvalidAppointmentInput, match="tenant_id is required"):
        await service.start_request(
            tenant_id=empty_val,
            conversation_id=CONVERSATION_ID,
        )

    with pytest.raises(InvalidAppointmentInput, match="conversation_id is required"):
        await service.start_request(tenant_id="tenant-1", conversation_id=empty_val)


@pytest.mark.anyio
async def test_start_request_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_or_create_request.side_effect = RuntimeError("db down")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to create appointment request"):
        await service.start_request(
            tenant_id="tenant-1",
            conversation_id=CONVERSATION_ID,
        )


@pytest.mark.anyio
async def test_get_request_returns_request_when_found() -> None:
    store = AsyncMock()
    expected = make_request()
    store.get_request.return_value = expected
    service = AppointmentService(store=store)

    result = await service.get_request(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert result == expected
    store.get_request.assert_awaited_once_with(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )


@pytest.mark.anyio
async def test_get_request_raises_not_found_when_none() -> None:
    store = AsyncMock()
    store.get_request.return_value = None
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentNotFound):
        await service.get_request(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_get_request_validates_required_fields() -> None:
    store = AsyncMock()
    service = AppointmentService(store=store)

    with pytest.raises(
        InvalidAppointmentInput, match="appointment_request_id is required"
    ):
        await service.get_request(appointment_request_id="", tenant_id="tenant-1")

    with pytest.raises(InvalidAppointmentInput, match="tenant_id is required"):
        await service.get_request(appointment_request_id=REQUEST_ID, tenant_id="   ")


@pytest.mark.anyio
async def test_get_request_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_request.side_effect = RuntimeError("query error")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to load appointment request"):
        await service.get_request(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_update_details_updates_fields_and_persists() -> None:
    store = AsyncMock()
    initial = make_request(status="collecting", department=None, consent_given=False)
    store.get_request.return_value = initial
    store.update_request.side_effect = lambda req: req
    service = AppointmentService(store=store)

    updated = await service.update_details(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        department="Neurology",
        preferred_date="2026-11-25",
        preferred_time="02:00 PM",
        contact="+1-555-0200",
        reason="Headache evaluation",
        consent_given=True,
    )

    assert updated.department == "Neurology"
    assert updated.preferred_date == "2026-11-25"
    assert updated.preferred_time == "02:00 PM"
    assert updated.contact == "+1-555-0200"
    assert updated.reason == "Headache evaluation"
    assert updated.consent_given is True
    store.update_request.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status", ["awaiting_review", "approved", "rejected", "booked"]
)
async def test_update_details_conflict_when_not_collecting(status: str) -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status=status)
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentConflict, match="cannot be changed from"):
        await service.update_details(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
            department="Neurology",
        )


@pytest.mark.anyio
async def test_update_details_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="collecting")
    store.update_request.side_effect = RuntimeError("update error")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to update appointment request"):
        await service.update_details(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
            department="Neurology",
        )


def test_missing_fields_identifies_unfilled() -> None:
    service = AppointmentService(store=AsyncMock())
    empty_req = make_request(
        department=None,
        preferred_date=None,
        contact=None,
        consent_given=False,
    )

    missing = service.missing_fields(empty_req)
    assert missing == ["department", "preferred_date", "contact", "consent_given"]

    complete_req = make_request(
        department="Oncology",
        preferred_date="2026-12-01",
        contact="user@test.org",
        consent_given=True,
    )
    assert service.missing_fields(complete_req) == []


@pytest.mark.anyio
async def test_submit_for_review_success() -> None:
    store = AsyncMock()
    req = make_request(status="collecting")
    store.get_request.return_value = req
    store.update_request.side_effect = lambda r: r
    service = AppointmentService(store=store)

    submitted = await service.submit_for_review(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        review_case_id=REVIEW_CASE_ID,
    )

    assert submitted.status == "awaiting_review"
    assert submitted.review_case_id == REVIEW_CASE_ID
    store.update_request.assert_awaited_once()


@pytest.mark.anyio
async def test_submit_for_review_conflict_when_not_collecting() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="awaiting_review")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentConflict, match="cannot be changed from"):
        await service.submit_for_review(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
            review_case_id=REVIEW_CASE_ID,
        )


@pytest.mark.anyio
async def test_submit_for_review_raises_invalid_when_missing_fields() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(
        status="collecting",
        contact=None,
        consent_given=False,
    )
    service = AppointmentService(store=store)

    with pytest.raises(
        InvalidAppointmentInput, match="missing: contact, consent_given"
    ):
        await service.submit_for_review(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
            review_case_id=REVIEW_CASE_ID,
        )


@pytest.mark.anyio
async def test_submit_for_review_validates_review_case_id() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="collecting")
    service = AppointmentService(store=store)

    with pytest.raises(InvalidAppointmentInput, match="review_case_id is required"):
        await service.submit_for_review(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
            review_case_id="   ",
        )


@pytest.mark.anyio
async def test_submit_for_review_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="collecting")
    store.update_request.side_effect = RuntimeError("update failed")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to submit appointment request"):
        await service.submit_for_review(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
            review_case_id=REVIEW_CASE_ID,
        )


@pytest.mark.anyio
async def test_approve_success() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="awaiting_review")
    store.update_request.side_effect = lambda r: r
    service = AppointmentService(store=store)

    approved = await service.approve(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert approved.status == "approved"
    store.update_request.assert_awaited_once()


@pytest.mark.anyio
async def test_approve_conflict_when_not_awaiting_review() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="collecting")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentConflict, match="cannot be changed from collecting"):
        await service.approve(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_approve_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="awaiting_review")
    store.update_request.side_effect = RuntimeError("db error")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to approve appointment request"):
        await service.approve(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_reject_success() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="awaiting_review")
    store.update_request.side_effect = lambda r: r
    service = AppointmentService(store=store)

    rejected = await service.reject(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert rejected.status == "rejected"
    store.update_request.assert_awaited_once()


@pytest.mark.anyio
async def test_reject_conflict_when_not_awaiting_review() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="collecting")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentConflict, match="cannot be changed from collecting"):
        await service.reject(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_reject_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="awaiting_review")
    store.update_request.side_effect = RuntimeError("db error")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to reject appointment request"):
        await service.reject(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_book_success() -> None:
    store = AsyncMock()
    req = make_request(status="approved")
    store.get_request.return_value = req
    created_app = make_appointment()
    store.book_appointment.return_value = created_app
    service = AppointmentService(store=store)

    result = await service.book(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert result == created_app
    store.book_appointment.assert_awaited_once()
    call = store.book_appointment.await_args.kwargs
    assert call["appointment_request_id"] == REQUEST_ID
    assert call["tenant_id"] == "tenant-1"
    assert call["appointment_id"]
    assert isinstance(call["created_at"], datetime)


@pytest.mark.anyio
async def test_book_idempotent_when_already_exists() -> None:
    store = AsyncMock()
    existing_app = make_appointment()
    req = make_request(status="booked", appointment_id=APPOINTMENT_ID)
    store.get_request.return_value = req
    store.book_appointment.return_value = existing_app
    service = AppointmentService(store=store)

    result = await service.book(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert result == existing_app
    store.book_appointment.assert_awaited_once()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status", ["collecting", "awaiting_review", "rejected", "failed"]
)
async def test_book_conflict_when_not_approved(
    status: AppointmentRequestStatus,
) -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status=status)
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentConflict, match="cannot be booked from"):
        await service.book(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_book_incomplete_request_raises_invalid_input() -> None:
    store = AsyncMock()
    incomplete = make_request(status="approved", department=None)
    store.get_request.return_value = incomplete
    service = AppointmentService(store=store)

    with pytest.raises(
        InvalidAppointmentInput, match="Approved appointment request is incomplete"
    ):
        await service.book(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )


@pytest.mark.anyio
async def test_book_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_request.return_value = make_request(status="approved")
    store.book_appointment.side_effect = RuntimeError("booking failed")
    service = AppointmentService(store=store)

    with pytest.raises(AppointmentError, match="Unable to book appointment"):
        await service.book(
            appointment_request_id=REQUEST_ID,
            tenant_id="tenant-1",
        )
