from typing import Literal, TypedDict

AgentRoute = Literal[
    "emergency",
    "education",
    "hospital_navigation",
    "appointment_request",
    "clarification",
    "human_review",
]


class AppointmentState(TypedDict, total=False):
    tenant_id: str
    conversation_id: str
    run_id: str

    query: str
    previous_assistant_message: str | None

    route: AgentRoute

    appointment_request_id: str
    missing_appointment_fields: list[str]

    review_case_id: str
    review_decision: Literal["approved", "rejected"]

    appointment_id: str

    response: str
