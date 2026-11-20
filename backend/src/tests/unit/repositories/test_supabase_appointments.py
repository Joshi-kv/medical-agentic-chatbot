from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from repositories.supabase_appointments import (
    SupabaseAppointmentStore,
    _appointment_from_row,
    _appointment_status,
    _datetime,
    _first_row,
    _optional_string,
    _request_from_row,
    _request_status,
    _request_to_row,
)
from services.appointments import (
    Appointment,
    AppointmentRequest,
    AppointmentRequestStatus,
    AppointmentStatus,
)

REQUEST_ID = "11111111-1111-4111-8111-111111111111"
CONVERSATION_ID = "22222222-2222-4222-8222-222222222222"
APPOINTMENT_ID = "33333333-3333-4333-8333-333333333333"


def make_query(data: object) -> MagicMock:
    query = MagicMock()
    for method in ("insert", "select", "update", "eq", "limit", "order"):
        getattr(query, method).return_value = query
    query.execute.return_value = SimpleNamespace(data=data)
    return query


def sample_request_row(
    status: AppointmentRequestStatus = "collecting",
) -> dict[str, Any]:
    now_iso = "2026-11-12T12:00:00+00:00"
    return {
        "appointment_request_id": REQUEST_ID,
        "tenant_id": "tenant-1",
        "conversation_id": CONVERSATION_ID,
        "department": "Cardiology",
        "preferred_date": "2026-11-20",
        "preferred_time": "10:00 AM",
        "contact": "+1-555-0100",
        "reason": "Follow-up",
        "consent_given": True,
        "status": status,
        "review_case_id": None,
        "appointment_id": None,
        "error_message": None,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


def sample_appointment_row(
    status: AppointmentStatus = "scheduled",
) -> dict[str, Any]:
    now_iso = "2026-11-12T12:00:00+00:00"
    return {
        "appointment_id": APPOINTMENT_ID,
        "tenant_id": "tenant-1",
        "appointment_request_id": REQUEST_ID,
        "department": "Cardiology",
        "scheduled_date": "2026-11-20",
        "scheduled_time": "10:00 AM",
        "contact": "+1-555-0100",
        "status": status,
        "created_at": now_iso,
    }


@pytest.mark.anyio
async def test_get_or_create_request_calls_transactional_rpc() -> None:
    client = MagicMock()
    query = make_query([sample_request_row()])
    client.rpc.return_value = query

    store = SupabaseAppointmentStore(client=client)
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    request = AppointmentRequest(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        conversation_id=CONVERSATION_ID,
        department="Cardiology",
        preferred_date="2026-11-20",
        preferred_time="10:00 AM",
        contact="+1-555-0100",
        reason="Follow-up",
        consent_given=True,
        status="collecting",
        review_case_id=None,
        appointment_id=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )

    result = await store.get_or_create_request(request)

    assert isinstance(result, AppointmentRequest)
    assert result.appointment_request_id == REQUEST_ID
    client.rpc.assert_called_once_with(
        "get_or_create_appointment_request",
        {
            "p_appointment_request_id": REQUEST_ID,
            "p_tenant_id": "tenant-1",
            "p_conversation_id": CONVERSATION_ID,
            "p_created_at": now.isoformat(),
        },
    )


@pytest.mark.anyio
async def test_get_request_returns_request_when_found() -> None:
    client = MagicMock()
    query = make_query([sample_request_row()])
    client.table.return_value = query

    store = SupabaseAppointmentStore(client=client)
    result = await store.get_request(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert isinstance(result, AppointmentRequest)
    assert result.appointment_request_id == REQUEST_ID
    client.table.assert_called_once_with("appointment_requests")
    query.eq.assert_any_call("appointment_request_id", REQUEST_ID)
    query.eq.assert_any_call("tenant_id", "tenant-1")


@pytest.mark.anyio
async def test_get_request_returns_none_when_empty() -> None:
    client = MagicMock()
    query = make_query([])
    client.table.return_value = query

    store = SupabaseAppointmentStore(client=client)
    result = await store.get_request(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
    )

    assert result is None


@pytest.mark.anyio
async def test_update_request_updates_and_returns_model() -> None:
    client = MagicMock()
    query = make_query([sample_request_row(status="awaiting_review")])
    client.table.return_value = query

    store = SupabaseAppointmentStore(client=client)
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    request = AppointmentRequest(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        conversation_id=CONVERSATION_ID,
        department="Cardiology",
        preferred_date="2026-11-20",
        preferred_time="10:00 AM",
        contact="+1-555-0100",
        reason="Follow-up",
        consent_given=True,
        status="awaiting_review",
        review_case_id="case-1",
        appointment_id=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )

    result = await store.update_request(request)

    assert isinstance(result, AppointmentRequest)
    assert result.status == "awaiting_review"
    client.table.assert_called_once_with("appointment_requests")
    query.update.assert_called_once()
    query.eq.assert_any_call("appointment_request_id", REQUEST_ID)
    query.eq.assert_any_call("tenant_id", "tenant-1")


@pytest.mark.anyio
async def test_list_requests_with_status_filter() -> None:
    client = MagicMock()
    query = make_query([sample_request_row(status="collecting")])
    client.table.return_value = query

    store = SupabaseAppointmentStore(client=client)
    results = await store.list_requests(
        tenant_id="tenant-1",
        status="collecting",
        limit=10,
    )

    assert len(results) == 1
    assert results[0].appointment_request_id == REQUEST_ID
    query.eq.assert_any_call("tenant_id", "tenant-1")
    query.eq.assert_any_call("status", "collecting")
    query.limit.assert_called_once_with(10)


@pytest.mark.anyio
async def test_list_requests_without_status_filter() -> None:
    client = MagicMock()
    query = make_query([sample_request_row()])
    client.table.return_value = query

    store = SupabaseAppointmentStore(client=client)
    results = await store.list_requests(
        tenant_id="tenant-1",
        status=None,
        limit=25,
    )

    assert len(results) == 1
    query.limit.assert_called_once_with(25)


@pytest.mark.anyio
async def test_list_requests_handles_none_data() -> None:
    client = MagicMock()
    query = make_query(None)
    client.table.return_value = query

    store = SupabaseAppointmentStore(client=client)
    results = await store.list_requests(
        tenant_id="tenant-1",
        status=None,
        limit=10,
    )

    assert results == []


@pytest.mark.anyio
async def test_book_appointment_calls_transactional_rpc() -> None:
    client = MagicMock()
    query = make_query([sample_appointment_row()])
    client.rpc.return_value = query

    store = SupabaseAppointmentStore(client=client)
    created_at = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    result = await store.book_appointment(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        appointment_id=APPOINTMENT_ID,
        created_at=created_at,
    )

    assert isinstance(result, Appointment)
    assert result.appointment_id == APPOINTMENT_ID
    client.rpc.assert_called_once_with(
        "book_appointment_request",
        {
            "p_appointment_request_id": REQUEST_ID,
            "p_tenant_id": "tenant-1",
            "p_appointment_id": APPOINTMENT_ID,
            "p_created_at": created_at.isoformat(),
        },
    )


def test_parsers_and_validation_errors() -> None:
    with pytest.raises(RuntimeError, match="invalid appointment response"):
        _first_row([])

    with pytest.raises(RuntimeError, match="invalid appointment response"):
        _first_row("invalid")

    assert _first_row({"key": "val"}) == {"key": "val"}

    with pytest.raises(RuntimeError, match="invalid timestamp"):
        _datetime(12345)

    assert _optional_string(None) is None
    assert _optional_string("test") == "test"

    with pytest.raises(RuntimeError, match="invalid request status"):
        _request_status("unknown_status")

    assert _request_status("collecting") == "collecting"

    with pytest.raises(RuntimeError, match="invalid appointment status"):
        _appointment_status("unknown_status")

    assert _appointment_status("scheduled") == "scheduled"
    assert _appointment_status("cancelled") == "cancelled"


def test_row_conversions() -> None:
    now = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
    request = AppointmentRequest(
        appointment_request_id=REQUEST_ID,
        tenant_id="tenant-1",
        conversation_id=CONVERSATION_ID,
        department="Cardiology",
        preferred_date="2026-11-20",
        preferred_time=None,
        contact="+1-555-0100",
        reason=None,
        consent_given=True,
        status="collecting",
        review_case_id=None,
        appointment_id=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )
    row = _request_to_row(request)
    assert row["appointment_request_id"] == REQUEST_ID
    assert row["department"] == "Cardiology"
    assert row["preferred_time"] is None

    from_row = _request_from_row(row)
    assert from_row.appointment_request_id == REQUEST_ID
    assert from_row.department == "Cardiology"
    assert from_row.preferred_time is None
    assert from_row.consent_given is True

    app_row = sample_appointment_row()
    app = _appointment_from_row(app_row)
    assert app.appointment_id == APPOINTMENT_ID
    assert app.department == "Cardiology"
    assert app.status == "scheduled"
