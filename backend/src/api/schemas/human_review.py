from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ResolveHumanReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    resolution: dict[str, Any] = Field(default_factory=dict)


class PatientHumanReviewCaseResponse(BaseModel):
    case_id: UUID
    conversation_id: UUID
    run_id: UUID
    review_type: Literal["clinical", "scheduling"]
    status: Literal[
        "pending",
        "claimed",
        "resolved",
        "rejected",
        "expired",
    ]
    decision: Literal["approved", "rejected"] | None
    message: str | None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime


class HumanReviewCaseResponse(BaseModel):
    case_id: UUID
    conversation_id: UUID
    run_id: UUID
    review_type: Literal["clinical", "scheduling"]
    status: Literal[
        "pending",
        "claimed",
        "resolved",
        "rejected",
        "expired",
    ]
    requested_action: str
    context: dict[str, Any]
    claimed_by: str | None
    claimed_at: datetime | None
    decision: Literal["approved", "rejected"] | None
    resolution: dict[str, Any] | None
    resolved_by: str | None
    resolved_at: datetime | None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime
