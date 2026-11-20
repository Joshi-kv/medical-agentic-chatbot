from datetime import datetime
from typing import Any, cast

from anyio import to_thread
from supabase import Client

from services.appointments import (
    Appointment,
    AppointmentRequest,
    AppointmentRequestStatus,
    AppointmentStatus,
)

REQUEST_COLUMNS = (
    "appointment_request_id,tenant_id,conversation_id,department,"
    "preferred_date,preferred_time,contact,reason,consent_given,"
    "status,review_case_id,appointment_id,error_message,"
    "created_at,updated_at"
)

APPOINTMENT_COLUMNS = (
    "appointment_id,tenant_id,appointment_request_id,department,"
    "scheduled_date,scheduled_time,contact,status,created_at"
)


class SupabaseAppointmentStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def get_or_create_request(
        self,
        request: AppointmentRequest,
    ) -> AppointmentRequest:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "get_or_create_appointment_request",
                {
                    "p_appointment_request_id": request.appointment_request_id,
                    "p_tenant_id": request.tenant_id,
                    "p_conversation_id": request.conversation_id,
                    "p_created_at": request.created_at.isoformat(),
                },
            ).execute()
        )

        return _request_from_row(_first_row(response.data))

    async def get_request(
        self,
        appointment_request_id: str,
        tenant_id: str,
    ) -> AppointmentRequest | None:
        response = await to_thread.run_sync(
            lambda: (
                self._client.table("appointment_requests")
                .select(REQUEST_COLUMNS)
                .eq(
                    "appointment_request_id",
                    appointment_request_id,
                )
                .eq("tenant_id", tenant_id)
                .limit(1)
                .execute()
            )
        )

        if not response.data:
            return None

        return _request_from_row(_first_row(response.data))

    async def update_request(
        self,
        request: AppointmentRequest,
    ) -> AppointmentRequest:
        row = _request_to_row(request)

        response = await to_thread.run_sync(
            lambda: (
                self._client.table("appointment_requests")
                .update(row)
                .eq(
                    "appointment_request_id",
                    request.appointment_request_id,
                )
                .eq("tenant_id", request.tenant_id)
                .execute()
            )
        )

        return _request_from_row(_first_row(response.data))

    async def list_requests(
        self,
        tenant_id: str,
        status: AppointmentRequestStatus | None,
        limit: int,
    ) -> list[AppointmentRequest]:
        def execute_query():
            query = (
                self._client.table("appointment_requests")
                .select(REQUEST_COLUMNS)
                .eq("tenant_id", tenant_id)
            )

            if status is not None:
                query = query.eq("status", status)

            return query.order("created_at", desc=False).limit(limit).execute()

        response = await to_thread.run_sync(execute_query)

        return [_request_from_row(row) for row in response.data or []]

    async def book_appointment(
        self,
        appointment_request_id: str,
        tenant_id: str,
        appointment_id: str,
        created_at: datetime,
    ) -> Appointment:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "book_appointment_request",
                {
                    "p_appointment_request_id": appointment_request_id,
                    "p_tenant_id": tenant_id,
                    "p_appointment_id": appointment_id,
                    "p_created_at": created_at.isoformat(),
                },
            ).execute()
        )

        return _appointment_from_row(_first_row(response.data))


def _request_to_row(
    request: AppointmentRequest,
) -> dict[str, object]:
    return {
        "appointment_request_id": (request.appointment_request_id),
        "tenant_id": request.tenant_id,
        "conversation_id": request.conversation_id,
        "department": request.department,
        "preferred_date": request.preferred_date,
        "preferred_time": request.preferred_time,
        "contact": request.contact,
        "reason": request.reason,
        "consent_given": request.consent_given,
        "status": request.status,
        "review_case_id": request.review_case_id,
        "appointment_id": request.appointment_id,
        "error_message": request.error_message,
        "created_at": request.created_at.isoformat(),
        "updated_at": request.updated_at.isoformat(),
    }


def _request_from_row(
    row: dict[str, Any],
) -> AppointmentRequest:
    return AppointmentRequest(
        appointment_request_id=str(row["appointment_request_id"]),
        tenant_id=str(row["tenant_id"]),
        conversation_id=str(row["conversation_id"]),
        department=_optional_string(row.get("department")),
        preferred_date=_optional_string(row.get("preferred_date")),
        preferred_time=_optional_string(row.get("preferred_time")),
        contact=_optional_string(row.get("contact")),
        reason=_optional_string(row.get("reason")),
        consent_given=bool(row["consent_given"]),
        status=_request_status(row["status"]),
        review_case_id=_optional_string(row.get("review_case_id")),
        appointment_id=_optional_string(row.get("appointment_id")),
        error_message=_optional_string(row.get("error_message")),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _appointment_from_row(
    row: dict[str, Any],
) -> Appointment:
    return Appointment(
        appointment_id=str(row["appointment_id"]),
        tenant_id=str(row["tenant_id"]),
        appointment_request_id=str(row["appointment_request_id"]),
        department=str(row["department"]),
        scheduled_date=str(row["scheduled_date"]),
        scheduled_time=_optional_string(row.get("scheduled_time")),
        contact=str(row["contact"]),
        status=_appointment_status(row["status"]),
        created_at=_datetime(row["created_at"]),
    )


def _first_row(data: object) -> dict[str, Any]:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]

    if isinstance(data, dict):
        return data

    raise RuntimeError("Supabase returned an invalid appointment response.")


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError("Supabase returned an invalid timestamp.")

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_string(value: object) -> str | None:
    if value is None:
        return None

    return str(value)


def _request_status(
    value: object,
) -> AppointmentRequestStatus:
    status = str(value)

    allowed = {
        "collecting",
        "awaiting_review",
        "approved",
        "rejected",
        "booked",
        "failed",
    }

    if status not in allowed:
        raise RuntimeError("Supabase returned an invalid request status.")

    return cast(AppointmentRequestStatus, status)


def _appointment_status(
    value: object,
) -> AppointmentStatus:
    status = str(value)

    if status not in {"scheduled", "cancelled"}:
        raise RuntimeError("Supabase returned an invalid appointment status.")

    return cast(AppointmentStatus, status)
