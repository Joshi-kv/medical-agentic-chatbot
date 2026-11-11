from dataclasses import dataclass
from enum import StrEnum


class SafetyAction(StrEnum):
    CONTINUE = "continue"
    EMERGENCY = "emergency"


@dataclass(frozen=True)
class SafetyDecision:
    action: SafetyAction
    reason: str | None = None
    response: str | None = None


class AgentRoute(StrEnum):
    EDUCATION = "education"
    HOSPITAL_NAVIGATION = "hospital_navigation"
    APPOINTMENT_REQUEST = "appointment_request"
    CLARIFICATION = "clarification"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True)
class RouteDecision:
    route: AgentRoute
    confidence: float
    follow_up_question: str | None = None
