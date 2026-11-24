from typing import Literal

from langgraph.types import interrupt

from agent.workflows.appointments.state import AppointmentState
from services.appointment_extractor import AppointmentExtractor
from services.appointments import (
    AppointmentConflict,
    AppointmentRequest,
    AppointmentService,
)
from services.human_review import HumanReviewService

AppointmentNextStep = Literal[
    "ask_next_appointment_question",
    "request_appointment_review",
]


APPOINTMENT_QUESTIONS = {
    "department": ("Which hospital department or type of care do you need?"),
    "preferred_date": ("What date would you prefer for the appointment?"),
    "contact": ("What phone number or email should the hospital use to contact you?"),
    "consent_given": (
        "May I send these appointment details to hospital staff for review?"
    ),
}


class AppointmentNodes:
    def __init__(
        self,
        appointment_service: AppointmentService,
        extractor: AppointmentExtractor,
        human_review_service: HumanReviewService,
    ) -> None:
        self._appointment_service = appointment_service
        self._extractor = extractor
        self._human_review_service = human_review_service

    async def ensure_appointment_request(
        self, state: AppointmentState
    ) -> dict[str, object]:
        """Create appointment request on first turn"""

        existing_request_id = state.get("appointment_request_id")

        if existing_request_id:
            return {}

        request = await self._appointment_service.start_request(
            tenant_id=self._required_state(state, "tenant_id"),
            conversation_id=self._required_state(state, "conversation_id"),
        )

        return {
            "appointment_request_id": (request.appointment_request_id),
        }

    async def collect_appointment_details(
        self,
        state: AppointmentState,
    ) -> dict[str, object]:
        """Extract and save info from latest message."""
        tenant_id = self._required_state(state, "tenant_id")
        appointment_request_id = self._required_state(state, "appointment_request_id")
        patient_message = self._required_state(state, "query")

        request = await self._appointment_service.get_request(
            appointment_request_id=appointment_request_id, tenant_id=tenant_id
        )

        extracted = await self._extractor.extract(
            patient_message=patient_message,
            current_details=self._request_details(request=request),
            previous_assistant_message=state.get("previous_assistant_message"),
        )

        extracted_values = extracted.model_dump(exclude_none=True)

        if extracted_values:
            request = await self._appointment_service.update_details(
                appointment_request_id=appointment_request_id,
                tenant_id=tenant_id,
                department=extracted.department,
                preferred_date=extracted.preferred_date,
                preferred_time=extracted.preferred_time,
                contact=extracted.contact,
                reason=extracted.reason,
                consent_given=extracted.consent_given,
            )

        missing_fields = self._appointment_service.missing_fields(request)

        return {
            "missing_appointment_fields": missing_fields,
        }

    def choose_next_appointment_step(
        self,
        state: AppointmentState,
    ) -> AppointmentNextStep:
        """Choose weather to ask followup or ask for review"""

        missing_fields = state.get("missing_appointment_fields")

        if missing_fields is None:
            raise ValueError("Graph state is missing missing_appointment_fields.")

        if missing_fields:
            return "ask_next_appointment_question"

        return "request_appointment_review"

    async def ask_next_appointment_question(
        self,
        state: AppointmentState,
    ) -> dict[str, object]:
        """Ask next follow up question"""

        missing_fields = state.get("missing_appointment_fields")

        if missing_fields is None:
            raise ValueError("Graph state is missing missing_appointment_fields.")

        if not missing_fields:
            return {}

        next_field = missing_fields[0]
        question = APPOINTMENT_QUESTIONS.get(next_field)

        if question is None:
            raise ValueError(f"Unknown field {next_field}")

        return {"response": question, "previous_assistant_message": question}

    async def request_appointment_review(
        self,
        state: AppointmentState,
    ) -> dict[str, object]:
        """Pause the appointment workflow for staff review."""

        tenant_id = self._required_state(state, "tenant_id")

        conversation_id = self._required_state(state, "conversation_id")

        run_id = self._required_state(state, "run_id")

        appointment_request_id = self._required_state(state, "appointment_request_id")

        appointment_request = await self._appointment_service.get_request(
            appointment_request_id=appointment_request_id,
            tenant_id=tenant_id,
        )

        request = appointment_request

        if request.status == "collecting":
            review_case = await self._human_review_service.request_review(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
                query=(
                    f"Review the completed appointment request {appointment_request_id}"
                ),
                review_type="scheduling",
            )

            request = await self._appointment_service.submit_for_review(
                appointment_request_id=appointment_request_id,
                tenant_id=tenant_id,
                review_case_id=review_case.case_id,
            )

        elif request.status == "awaiting_review":
            pass

        else:
            raise AppointmentConflict(
                f"Appointment request cannot be reviewed from {request.status}."
            )

        case_id = request.review_case_id

        if case_id is None:
            raise AppointmentConflict("Appointment request has no review case.")

        resume_value = interrupt(
            {
                "type": "appointment_review",
                "case_id": case_id,
                "appointment_request_id": appointment_request_id,
                "message": (
                    "Your appointment request has been sent "
                    "to hospital staff for review."
                ),
            }
        )

        if not isinstance(resume_value, dict):
            raise ValueError("Human-review resume value must be an object.")

        decision = resume_value.get("decision")

        if decision not in {"approved", "rejected"}:
            raise ValueError("Human-review decision must be approved or rejected.")

        return {
            "review_case_id": case_id,
            "review_decision": decision,
        }

    def choose_after_appointment_review(
        self,
        state: AppointmentState,
    ) -> Literal[
        "book_appointment",
        "appointment_rejected",
    ]:
        """Route a staff decision to booking or rejection."""
        decision = state.get("review_decision")

        if decision == "approved":
            return "book_appointment"

        if decision == "rejected":
            return "appointment_rejected"

        raise ValueError("Graph state is missing a valid review_decision.")

    async def book_appointment(
        self,
        state: AppointmentState,
    ) -> dict[str, object]:
        """Book an appointment after staff approval."""
        if state.get("review_decision") != "approved":
            raise ValueError("An approved review decision is required before booking.")

        tenant_id = self._required_state(state, "tenant_id")
        request_id = self._required_state(
            state,
            "appointment_request_id",
        )
        request = await self._appointment_service.get_request(
            appointment_request_id=request_id,
            tenant_id=tenant_id,
        )

        if request.status == "awaiting_review":
            await self._appointment_service.approve(
                appointment_request_id=request_id,
                tenant_id=tenant_id,
            )
        elif request.status not in {"approved", "booked"}:
            raise AppointmentConflict(
                f"Appointment request cannot be booked from {request.status}."
            )

        appointment = await self._appointment_service.book(
            appointment_request_id=request_id,
            tenant_id=tenant_id,
        )
        message = (
            "Your appointment request has been approved and scheduled. "
            "Hospital staff will use the contact information you provided "
            "for confirmation."
        )

        return {
            "appointment_id": appointment.appointment_id,
            "response": message,
            "previous_assistant_message": message,
        }

    async def appointment_rejected(
        self,
        state: AppointmentState,
    ) -> dict[str, object]:
        """Record a rejected appointment request."""
        if state.get("review_decision") != "rejected":
            raise ValueError("A rejected review decision is required.")

        tenant_id = self._required_state(state, "tenant_id")
        request_id = self._required_state(
            state,
            "appointment_request_id",
        )
        request = await self._appointment_service.get_request(
            appointment_request_id=request_id,
            tenant_id=tenant_id,
        )

        if request.status == "awaiting_review":
            await self._appointment_service.reject(
                appointment_request_id=request_id,
                tenant_id=tenant_id,
            )
        elif request.status != "rejected":
            raise AppointmentConflict(
                f"Appointment request cannot be rejected from {request.status}."
            )

        message = (
            "Hospital staff could not approve this appointment request. "
            "Please contact the hospital if you need more information "
            "or want to submit another request."
        )

        return {
            "response": message,
            "previous_assistant_message": message,
        }

    def _request_details(
        self,
        request: AppointmentRequest,
    ) -> dict[str, object]:
        return {
            "department": request.department,
            "preferred_date": request.preferred_date,
            "preferred_time": request.preferred_time,
            "contact": request.contact,
            "reason": request.reason,
            "consent_given": request.consent_given,
        }

    def _required_state(
        self,
        state: AppointmentState,
        field_name: str,
    ) -> str:
        value = state.get(field_name)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Graph state is missing {field_name}.")

        return value.strip()
