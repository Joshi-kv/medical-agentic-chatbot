from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from services.human_review import (
    CreateHumanReview,
    HumanReviewCase,
    HumanReviewConflict,
    HumanReviewError,
    HumanReviewNotFound,
    HumanReviewService,
    InvalidHumanReviewInput,
)

CASE_ID = "11111111-1111-4111-8111-111111111111"
CONVERSATION_ID = "22222222-2222-4222-8222-222222222222"
RUN_ID = "33333333-3333-4333-8333-333333333333"


def make_case(
    case_id: str = CASE_ID,
    tenant_id: str = "tenant-1",
    conversation_id: str = CONVERSATION_ID,
    run_id: str = RUN_ID,
    status: str = "pending",
    claimed_by: str | None = None,
    decision: str | None = None,
) -> HumanReviewCase:
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    return HumanReviewCase(
        case_id=case_id,
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        run_id=run_id,
        review_type="clinical",
        status=status,  # type: ignore[arg-type]
        requested_action="Review emergency dosage",
        context={"query": "Review emergency dosage", "route": "human_review"},
        claimed_by=claimed_by,
        claimed_at=now if claimed_by else None,
        decision=decision,  # type: ignore[arg-type]
        resolution={"note": "Approved by MD"} if decision else None,
        resolved_by="staff-1" if decision else None,
        resolved_at=now if decision else None,
        expires_at=now + timedelta(hours=24),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.anyio
async def test_request_review_creates_and_returns_case() -> None:
    store = AsyncMock()
    case = make_case()
    store.create_case.return_value = case
    service = HumanReviewService(store=store)

    result = await service.request_review(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        run_id=RUN_ID,
        query="Review emergency dosage",
        review_type="clinical",
    )

    assert result == case
    store.create_case.assert_awaited_once()
    command = store.create_case.await_args.args[0]
    assert isinstance(command, CreateHumanReview)
    assert command.conversation_id == CONVERSATION_ID
    assert command.tenant_id == "tenant-1"
    assert command.run_id == RUN_ID
    assert command.review_type == "clinical"
    assert command.requested_action == "review_user_request"
    assert command.context["query"] == "Review emergency dosage"
    assert command.expires_at > command.created_at


@pytest.mark.anyio
@pytest.mark.parametrize("empty_query", ["", "   ", "\n\t"])
async def test_request_review_rejects_empty_query(empty_query: str) -> None:
    store = AsyncMock()
    service = HumanReviewService(store=store)

    with pytest.raises(InvalidHumanReviewInput, match="query cannot be empty"):
        await service.request_review(
            conversation_id=CONVERSATION_ID,
            tenant_id="tenant-1",
            run_id=RUN_ID,
            query=empty_query,
        )

    store.create_case.assert_not_awaited()


@pytest.mark.anyio
async def test_request_review_wraps_store_error() -> None:
    store = AsyncMock()
    store.create_case.side_effect = RuntimeError("database down")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewError, match="Could not request human review"):
        await service.request_review(
            conversation_id=CONVERSATION_ID,
            tenant_id="tenant-1",
            run_id=RUN_ID,
            query="Review dosage",
        )


@pytest.mark.anyio
async def test_get_case_returns_record() -> None:
    store = AsyncMock()
    case = make_case()
    store.get_case.return_value = case
    service = HumanReviewService(store=store)

    result = await service.get(case_id=CASE_ID, tenant_id="tenant-1")

    assert result == case
    store.expire_cases.assert_awaited_once_with("tenant-1")
    store.get_case.assert_awaited_once_with(case_id=CASE_ID, tenant_id="tenant-1")


@pytest.mark.anyio
async def test_get_case_raises_not_found() -> None:
    store = AsyncMock()
    store.get_case.return_value = None
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewNotFound):
        await service.get(case_id=CASE_ID, tenant_id="tenant-1")


@pytest.mark.anyio
async def test_get_case_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_case.side_effect = RuntimeError("database down")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewError, match="Unable to load review case"):
        await service.get(case_id=CASE_ID, tenant_id="tenant-1")


@pytest.mark.anyio
async def test_list_cases_forwards_parameters() -> None:
    store = AsyncMock()
    case = make_case()
    store.list_cases.return_value = [case]
    service = HumanReviewService(store=store)

    results = await service.list_cases(
        tenant_id="tenant-1",
        status="pending",
        limit=25,
    )

    assert results == [case]
    store.expire_cases.assert_awaited_once_with("tenant-1")
    store.list_cases.assert_awaited_once_with(
        tenant_id="tenant-1",
        status="pending",
        limit=25,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("invalid_limit", [0, -1, 101, 500])
async def test_list_cases_rejects_invalid_limit(invalid_limit: int) -> None:
    store = AsyncMock()
    service = HumanReviewService(store=store)

    with pytest.raises(
        InvalidHumanReviewInput, match="limit must be between 1 and 100"
    ):
        await service.list_cases(tenant_id="tenant-1", limit=invalid_limit)


@pytest.mark.anyio
async def test_claim_case_success() -> None:
    store = AsyncMock()
    claimed = make_case(status="claimed", claimed_by="staff-1")
    store.claim_case.return_value = claimed
    service = HumanReviewService(store=store)

    result = await service.claim(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
    )

    assert result == claimed
    store.claim_case.assert_awaited_once_with(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
    )


@pytest.mark.anyio
async def test_claim_case_expired_raises_conflict() -> None:
    store = AsyncMock()
    store.claim_case.return_value = None
    store.get_case.return_value = make_case(status="expired")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewConflict, match="expired"):
        await service.claim(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
        )


@pytest.mark.anyio
async def test_claim_case_already_claimed_raises_conflict() -> None:
    store = AsyncMock()
    store.claim_case.return_value = None
    store.get_case.return_value = make_case(status="claimed", claimed_by="staff-2")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewConflict, match="claimed by another staff member"):
        await service.claim(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
        )


@pytest.mark.anyio
async def test_claim_case_from_resolved_raises_conflict() -> None:
    store = AsyncMock()
    store.claim_case.return_value = None
    store.get_case.return_value = make_case(status="resolved", decision="approved")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewConflict, match="cannot be claimed from resolved"):
        await service.claim(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
        )


@pytest.mark.anyio
async def test_resolve_case_success() -> None:
    store = AsyncMock()
    resolved = make_case(status="resolved", decision="approved")
    store.resolve_case.return_value = resolved
    service = HumanReviewService(store=store)

    result = await service.resolve(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={"note": "Looks accurate"},
    )

    assert result == resolved
    store.resolve_case.assert_awaited_once_with(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={"note": "Looks accurate"},
    )


@pytest.mark.anyio
async def test_resolve_case_rejects_invalid_decision() -> None:
    store = AsyncMock()
    service = HumanReviewService(store=store)

    with pytest.raises(InvalidHumanReviewInput, match="Invalid review decision"):
        await service.resolve(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
            decision="invalid_decision",  # type: ignore[arg-type]
            resolution={},
        )


@pytest.mark.anyio
async def test_resolve_case_rejects_non_dict_resolution() -> None:
    store = AsyncMock()
    service = HumanReviewService(store=store)

    with pytest.raises(InvalidHumanReviewInput, match="resolution must be an object"):
        await service.resolve(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
            decision="approved",
            resolution="not-a-dict",  # type: ignore[arg-type]
        )


@pytest.mark.anyio
async def test_resolve_case_conflict_when_claimed_by_other() -> None:
    store = AsyncMock()
    store.resolve_case.return_value = None
    store.get_case.return_value = make_case(status="claimed", claimed_by="staff-2")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewConflict, match="Only the staff member who claimed"):
        await service.resolve(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
            decision="approved",
            resolution={},
        )


@pytest.mark.anyio
async def test_resolve_case_conflict_from_other_status() -> None:
    store = AsyncMock()
    store.resolve_case.return_value = None
    store.get_case.return_value = make_case(status="pending")
    service = HumanReviewService(store=store)

    with pytest.raises(HumanReviewConflict, match="cannot be resolved from pending"):
        await service.resolve(
            case_id=CASE_ID,
            tenant_id="tenant-1",
            staff_id="staff-1",
            decision="approved",
            resolution={},
        )
