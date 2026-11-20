from services.appointment_workflow import AppointmentWorkflow
from services.human_review import (
    HumanReviewCase,
    HumanReviewConflict,
    HumanReviewError,
    HumanReviewService,
    InvalidHumanReviewInput,
    ReviewDecision,
)


class ReviewResolutionService:
    def __init__(
        self,
        human_reviews: HumanReviewService,
        appointment_workflow: AppointmentWorkflow,
    ) -> None:
        self._human_reviews = human_reviews
        self._appointment_workflow = appointment_workflow

    async def resolve(
        self,
        tenant_id: str,
        case_id: str,
        staff_id: str,
        decision: ReviewDecision,
        resolution: dict[str, object],
    ) -> HumanReviewCase:
        if decision not in {"approved", "rejected"}:
            raise InvalidHumanReviewInput("Invalid review decision.")

        if not isinstance(resolution, dict):
            raise InvalidHumanReviewInput("resolution must be an object.")

        case = await self._human_reviews.get(tenant_id=tenant_id, case_id=case_id)

        if case.review_type == "clinical":
            return await self._human_reviews.resolve(
                case_id=case_id,
                tenant_id=tenant_id,
                staff_id=staff_id,
                decision=decision,
                resolution=resolution,
            )

        self._validate_scheduling_resolution(
            case=case,
            staff_id=staff_id,
            decision=decision,
        )

        try:
            result = await self._appointment_workflow.resume(
                tenant_id=tenant_id,
                conversation_id=case.conversation_id,
                case_id=case_id,
                decision=decision,
            )
        except Exception as error:
            raise HumanReviewError("Unable to resume appointment workflow.") from error

        completed_resolution = {**resolution, "message": result.answer}

        return await self._human_reviews.resolve(
            case_id=case_id,
            tenant_id=tenant_id,
            staff_id=staff_id,
            decision=decision,
            resolution=completed_resolution,
        )

    def _validate_scheduling_resolution(
        self,
        case: HumanReviewCase,
        staff_id: str,
        decision: ReviewDecision,
    ) -> None:
        if (
            case.status in {"resolved", "rejected"}
            and case.resolved_by == staff_id
            and case.decision == decision
        ):
            return

        if case.status != "claimed":
            raise HumanReviewConflict(
                f"Review case cannot be resolved from {case.status}."
            )

        if case.claimed_by != staff_id:
            raise HumanReviewConflict(
                "Only the staff member who claimed the case can resolve it."
            )
