from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol
from uuid import uuid4

ReviewType = Literal["clinical", "scheduling"]
ReviewStatus = Literal[
    "pending",
    "claimed",
    "resolved",
    "rejected",
    "expired",
]
ReviewDecision = Literal["approved", "rejected"]

DEFAULT_REVIEW_TIMEOUT = timedelta(hours=24)


class HumanReviewError(Exception):
    pass


class HumanReviewNotFound(HumanReviewError):
    pass


class HumanReviewConflict(HumanReviewError):
    pass


class InvalidHumanReviewInput(HumanReviewError):
    pass


@dataclass(frozen=True)
class HumanReviewCase:
    case_id: str
    tenant_id: str
    conversation_id: str
    run_id: str
    review_type: ReviewType
    status: ReviewStatus
    requested_action: str
    context: dict[str, Any]
    claimed_by: str | None
    claimed_at: datetime | None
    decision: ReviewDecision | None
    resolution: dict[str, Any] | None
    resolved_by: str | None
    resolved_at: datetime | None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class CreateHumanReview:
    case_id: str
    tenant_id: str
    conversation_id: str
    run_id: str
    review_type: ReviewType
    requested_action: str
    context: dict[str, Any]
    expires_at: datetime
    created_at: datetime


class HumanReviewStore(Protocol):
    async def expire_cases(self, tenant_id: str) -> int: ...

    async def create_case(
        self,
        command: CreateHumanReview,
    ) -> HumanReviewCase: ...

    async def get_case(
        self,
        case_id: str,
        tenant_id: str,
    ) -> HumanReviewCase | None: ...

    async def list_cases(
        self,
        tenant_id: str,
        status: ReviewStatus | None,
        limit: int,
    ) -> list[HumanReviewCase]: ...

    async def claim_case(
        self,
        case_id: str,
        tenant_id: str,
        staff_id: str,
    ) -> HumanReviewCase | None: ...

    async def resolve_case(
        self,
        case_id: str,
        tenant_id: str,
        staff_id: str,
        decision: ReviewDecision,
        resolution: dict[str, Any],
    ) -> HumanReviewCase | None: ...


class HumanReviewService:
    def __init__(self, store: HumanReviewStore) -> None:
        self._store = store

    async def request_review(
        self,
        conversation_id: str,
        tenant_id: str,
        run_id: str,
        query: str,
        review_type: ReviewType = "clinical",
    ) -> HumanReviewCase:

        normalized_query = query.strip()

        if not normalized_query:
            raise InvalidHumanReviewInput("Review query cannot be empty")

        now = datetime.now(timezone.utc)

        command = CreateHumanReview(
            case_id=str(uuid4()),
            tenant_id=self._required(tenant_id, "tenant_id"),
            conversation_id=self._required(conversation_id, "conversation_id"),
            run_id=self._required(run_id, "run_id"),
            review_type=review_type,
            requested_action="review_user_request",
            context={"query": normalized_query, "route": "human_review"},
            expires_at=now + DEFAULT_REVIEW_TIMEOUT,
            created_at=now,
        )

        try:
            return await self._store.create_case(command)
        except Exception as error:
            raise HumanReviewError("Could not request human review") from error

    async def get(
        self,
        case_id: str,
        tenant_id: str,
    ) -> HumanReviewCase:
        normalized_tenant_id = self._required(tenant_id, "tenant_id")

        try:
            await self._store.expire_cases(normalized_tenant_id)
            case = await self._store.get_case(
                case_id=self._required(case_id, "case_id"),
                tenant_id=normalized_tenant_id,
            )
        except Exception as error:
            raise HumanReviewError("Unable to load review case.") from error

        if case is None:
            raise HumanReviewNotFound

        return case

    async def list_cases(
        self,
        tenant_id: str,
        status: ReviewStatus | None = None,
        limit: int = 50,
    ) -> list[HumanReviewCase]:
        if limit < 1 or limit > 100:
            raise InvalidHumanReviewInput("limit must be between 1 and 100.")

        normalized_tenant_id = self._required(tenant_id, "tenant_id")

        try:
            await self._store.expire_cases(normalized_tenant_id)
            return await self._store.list_cases(
                tenant_id=normalized_tenant_id,
                status=status,
                limit=limit,
            )
        except Exception as error:
            raise HumanReviewError("Unable to list review cases.") from error

    async def claim(
        self,
        case_id: str,
        tenant_id: str,
        staff_id: str,
    ) -> HumanReviewCase:
        try:
            case = await self._store.claim_case(
                case_id=self._required(case_id, "case_id"),
                tenant_id=self._required(tenant_id, "tenant_id"),
                staff_id=self._required(staff_id, "staff_id"),
            )
        except Exception as error:
            raise HumanReviewError("Unable to claim review case.") from error

        if case is None:
            existing = await self.get(case_id, tenant_id)

            if existing.status == "expired":
                raise HumanReviewConflict("Review case has expired.")

            if existing.status == "claimed":
                raise HumanReviewConflict(
                    "Review case was claimed by another staff member."
                )

            raise HumanReviewConflict(
                f"Review case cannot be claimed from {existing.status}."
            )

        return case

    async def resolve(
        self,
        case_id: str,
        tenant_id: str,
        staff_id: str,
        decision: ReviewDecision,
        resolution: dict[str, Any],
    ) -> HumanReviewCase:
        if decision not in {"approved", "rejected"}:
            raise InvalidHumanReviewInput("Invalid review decision.")

        if not isinstance(resolution, dict):
            raise InvalidHumanReviewInput("resolution must be an object.")

        try:
            case = await self._store.resolve_case(
                case_id=self._required(case_id, "case_id"),
                tenant_id=self._required(tenant_id, "tenant_id"),
                staff_id=self._required(staff_id, "staff_id"),
                decision=decision,
                resolution=resolution,
            )
        except Exception as error:
            raise HumanReviewError("Unable to resolve review case.") from error

        if case is None:
            existing = await self.get(case_id, tenant_id)

            if existing.status == "claimed":
                raise HumanReviewConflict(
                    "Only the staff member who claimed the case can resolve it."
                )

            raise HumanReviewConflict(
                f"Review case cannot be resolved from {existing.status}."
            )

        return case

    def _required(self, value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise InvalidHumanReviewInput(f"{field_name} is required.")
        return normalized
