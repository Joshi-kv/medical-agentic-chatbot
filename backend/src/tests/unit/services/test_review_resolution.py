from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from services.answering import AnswerResult
from services.human_review import (
    HumanReviewCase,
    HumanReviewConflict,
    HumanReviewError,
)
from services.review_resolution import ReviewResolutionService


def make_case(
    review_type: str = "scheduling",
    status: str = "claimed",
    claimed_by: str | None = "staff-1",
    decision: str | None = None,
    resolved_by: str | None = None,
) -> HumanReviewCase:
    now = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)
    return HumanReviewCase(
        case_id="case-1",
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        run_id="run-1",
        review_type=review_type,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        requested_action="review_user_request",
        context={},
        claimed_by=claimed_by,
        claimed_at=now if claimed_by else None,
        decision=decision,  # type: ignore[arg-type]
        resolution=None,
        resolved_by=resolved_by,
        resolved_at=now if resolved_by else None,
        expires_at=now + timedelta(hours=24),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.anyio
async def test_scheduling_review_resumes_then_resolves_with_message() -> None:
    human_reviews = AsyncMock()
    workflow = AsyncMock()
    claimed = make_case()
    resolved = make_case(
        status="resolved",
        decision="approved",
        resolved_by="staff-1",
    )
    human_reviews.get.return_value = claimed
    human_reviews.resolve.return_value = resolved
    workflow.resume.return_value = AnswerResult(
        answer="Your appointment has been scheduled.",
        citations=[],
        sufficient_evidence=False,
        safety_notice="Educational only.",
        route="appointment_request",
        review_case_id="case-1",
    )
    service = ReviewResolutionService(human_reviews, workflow)

    result = await service.resolve(
        case_id="case-1",
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={"note": "Approved by scheduling"},
    )

    assert result is resolved
    workflow.resume.assert_awaited_once_with(
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        case_id="case-1",
        decision="approved",
    )
    human_reviews.resolve.assert_awaited_once_with(
        case_id="case-1",
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={
            "note": "Approved by scheduling",
            "message": "Your appointment has been scheduled.",
        },
    )


@pytest.mark.anyio
async def test_clinical_review_does_not_resume_appointment_graph() -> None:
    human_reviews = AsyncMock()
    workflow = AsyncMock()
    case = make_case(review_type="clinical")
    human_reviews.get.return_value = case
    human_reviews.resolve.return_value = case
    service = ReviewResolutionService(human_reviews, workflow)

    result = await service.resolve(
        case_id="case-1",
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={},
    )

    assert result is case
    workflow.resume.assert_not_awaited()
    human_reviews.resolve.assert_awaited_once()


@pytest.mark.anyio
async def test_wrong_staff_cannot_resume_scheduling_review() -> None:
    human_reviews = AsyncMock()
    workflow = AsyncMock()
    human_reviews.get.return_value = make_case(claimed_by="staff-2")
    service = ReviewResolutionService(human_reviews, workflow)

    with pytest.raises(HumanReviewConflict, match="claimed"):
        await service.resolve(
            case_id="case-1",
            tenant_id="tenant-1",
            staff_id="staff-1",
            decision="approved",
            resolution={},
        )

    workflow.resume.assert_not_awaited()
    human_reviews.resolve.assert_not_awaited()


@pytest.mark.anyio
async def test_workflow_failure_keeps_review_case_unresolved() -> None:
    human_reviews = AsyncMock()
    workflow = AsyncMock()
    human_reviews.get.return_value = make_case()
    workflow.resume.side_effect = RuntimeError("database unavailable")
    service = ReviewResolutionService(human_reviews, workflow)

    with pytest.raises(HumanReviewError, match="Unable to resume"):
        await service.resolve(
            case_id="case-1",
            tenant_id="tenant-1",
            staff_id="staff-1",
            decision="approved",
            resolution={},
        )

    human_reviews.resolve.assert_not_awaited()
