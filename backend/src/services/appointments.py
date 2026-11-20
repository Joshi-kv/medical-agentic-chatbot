from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

AppointmentRequestStatus = Literal[
    "collecting",
    "awaiting_review",
    "approved",
    "rejected",
    "booked",
    "failed",
]

AppointmentStatus = Literal["scheduled", "cancelled"]


class AppointmentError(Exception):
    """Base exception for appointment operations."""


class AppointmentNotFound(AppointmentError):
    """Raised when an appointment request cannot be found."""


class AppointmentConflict(AppointmentError):
    """Raised when an operation is invalid for the current status."""


class InvalidAppointmentInput(AppointmentError):
    """Raised when required appointment information is invalid."""


@dataclass(frozen=True)
class AppointmentRequest:
    appointment_request_id: str
    tenant_id: str
    conversation_id: str

    department: str | None
    preferred_date: str | None
    preferred_time: str | None
    contact: str | None
    reason: str | None
    consent_given: bool

    status: AppointmentRequestStatus
    review_case_id: str | None
    appointment_id: str | None
    error_message: str | None

    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Appointment:
    appointment_id: str
    tenant_id: str
    appointment_request_id: str

    department: str
    scheduled_date: str
    scheduled_time: str | None
    contact: str

    status: AppointmentStatus
    created_at: datetime


class AppointmentStore(Protocol):
    async def get_or_create_request(
        self,
        request: AppointmentRequest,
    ) -> AppointmentRequest: ...

    async def get_request(
        self,
        appointment_request_id: str,
        tenant_id: str,
    ) -> AppointmentRequest | None: ...

    async def update_request(
        self,
        request: AppointmentRequest,
    ) -> AppointmentRequest: ...

    async def list_requests(
        self,
        tenant_id: str,
        status: AppointmentRequestStatus | None,
        limit: int,
    ) -> list[AppointmentRequest]: ...

    async def book_appointment(
        self,
        appointment_request_id: str,
        tenant_id: str,
        appointment_id: str,
        created_at: datetime,
    ) -> Appointment: ...


class AppointmentService:
    def __init__(self, store: AppointmentStore) -> None:
        self._store = store

    async def start_request(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> AppointmentRequest:
        """Start collecting information for an appointment request."""
        now = datetime.now(timezone.utc)

        request = AppointmentRequest(
            appointment_request_id=str(uuid4()),
            tenant_id=self._required(tenant_id, "tenant_id"),
            conversation_id=self._required(
                conversation_id,
                "conversation_id",
            ),
            department=None,
            preferred_date=None,
            preferred_time=None,
            contact=None,
            reason=None,
            consent_given=False,
            status="collecting",
            review_case_id=None,
            appointment_id=None,
            error_message=None,
            created_at=now,
            updated_at=now,
        )

        try:
            return await self._store.get_or_create_request(request)
        except Exception as error:
            raise AppointmentError("Unable to create appointment request.") from error

    async def get_request(
        self,
        appointment_request_id: str,
        tenant_id: str,
    ) -> AppointmentRequest:
        """Return an appointment request belonging to a tenant."""
        request_id = self._required(
            appointment_request_id,
            "appointment_request_id",
        )
        normalized_tenant_id = self._required(
            tenant_id,
            "tenant_id",
        )

        try:
            request = await self._store.get_request(
                appointment_request_id=request_id,
                tenant_id=normalized_tenant_id,
            )
        except Exception as error:
            raise AppointmentError("Unable to load appointment request.") from error

        if request is None:
            raise AppointmentNotFound

        return request

    async def update_details(
        self,
        appointment_request_id: str,
        tenant_id: str,
        department: str | None = None,
        preferred_date: str | None = None,
        preferred_time: str | None = None,
        contact: str | None = None,
        reason: str | None = None,
        consent_given: bool | None = None,
    ) -> AppointmentRequest:
        """Add patient-provided information to a collecting request."""
        request = await self.get_request(
            appointment_request_id=appointment_request_id,
            tenant_id=tenant_id,
        )

        self._require_status(request, "collecting")

        updated = replace(
            request,
            department=(
                self._optional(department)
                if department is not None
                else request.department
            ),
            preferred_date=(
                self._optional(preferred_date)
                if preferred_date is not None
                else request.preferred_date
            ),
            preferred_time=(
                self._optional(preferred_time)
                if preferred_time is not None
                else request.preferred_time
            ),
            contact=(
                self._optional(contact) if contact is not None else request.contact
            ),
            reason=(self._optional(reason) if reason is not None else request.reason),
            consent_given=(
                consent_given if consent_given is not None else request.consent_given
            ),
            updated_at=datetime.now(timezone.utc),
        )

        try:
            return await self._store.update_request(updated)
        except Exception as error:
            raise AppointmentError("Unable to update appointment request.") from error

    def missing_fields(
        self,
        request: AppointmentRequest,
    ) -> list[str]:
        """Return information still required before staff review."""
        missing: list[str] = []

        if not request.department:
            missing.append("department")

        if not request.preferred_date:
            missing.append("preferred_date")

        if not request.contact:
            missing.append("contact")

        if request.consent_given is not True:
            missing.append("consent_given")

        return missing

    async def submit_for_review(
        self,
        appointment_request_id: str,
        tenant_id: str,
        review_case_id: str,
    ) -> AppointmentRequest:
        """Connect a complete request to a human-review case."""
        request = await self.get_request(
            appointment_request_id=appointment_request_id,
            tenant_id=tenant_id,
        )

        self._require_status(request, "collecting")

        missing = self.missing_fields(request)
        if missing:
            fields = ", ".join(missing)
            raise InvalidAppointmentInput(f"Appointment request is missing: {fields}.")

        updated = replace(
            request,
            status="awaiting_review",
            review_case_id=self._required(
                review_case_id,
                "review_case_id",
            ),
            updated_at=datetime.now(timezone.utc),
        )

        try:
            return await self._store.update_request(updated)
        except Exception as error:
            raise AppointmentError(
                "Unable to submit appointment request for review."
            ) from error

    async def approve(
        self,
        appointment_request_id: str,
        tenant_id: str,
    ) -> AppointmentRequest:
        """Approve a request after staff review."""
        request = await self.get_request(
            appointment_request_id=appointment_request_id,
            tenant_id=tenant_id,
        )

        self._require_status(request, "awaiting_review")

        approved = replace(
            request,
            status="approved",
            updated_at=datetime.now(timezone.utc),
        )

        try:
            return await self._store.update_request(approved)
        except Exception as error:
            raise AppointmentError("Unable to approve appointment request.") from error

    async def reject(
        self,
        appointment_request_id: str,
        tenant_id: str,
    ) -> AppointmentRequest:
        """Reject a request after staff review."""
        request = await self.get_request(
            appointment_request_id=appointment_request_id,
            tenant_id=tenant_id,
        )

        self._require_status(request, "awaiting_review")

        rejected = replace(
            request,
            status="rejected",
            updated_at=datetime.now(timezone.utc),
        )

        try:
            return await self._store.update_request(rejected)
        except Exception as error:
            raise AppointmentError("Unable to reject appointment request.") from error

    async def book(
        self,
        appointment_request_id: str,
        tenant_id: str,
    ) -> Appointment:
        """Create one appointment from an approved request."""
        request = await self.get_request(
            appointment_request_id=appointment_request_id,
            tenant_id=tenant_id,
        )

        if request.status not in {"approved", "booked"}:
            raise AppointmentConflict(
                f"Appointment request cannot be booked from {request.status}."
            )

        missing = self.missing_fields(request)
        if missing:
            raise InvalidAppointmentInput("Approved appointment request is incomplete.")

        try:
            return await self._store.book_appointment(
                appointment_request_id=request.appointment_request_id,
                tenant_id=request.tenant_id,
                appointment_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
            )
        except Exception as error:
            raise AppointmentError("Unable to book appointment.") from error

    def _require_status(
        self,
        request: AppointmentRequest,
        expected: AppointmentRequestStatus,
    ) -> None:
        if request.status != expected:
            raise AppointmentConflict(
                f"Appointment request cannot be changed from "
                f"{request.status}; expected {expected}."
            )

    def _required(
        self,
        value: str,
        field_name: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise InvalidAppointmentInput(f"{field_name} is required.")

        return normalized

    def _optional(self, value: str) -> str | None:
        normalized = value.strip()
        return normalized or None
