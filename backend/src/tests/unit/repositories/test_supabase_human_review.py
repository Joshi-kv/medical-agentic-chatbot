from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from repositories.supabase_human_review import (
    SupabaseHumanReviewStore,
    _case_from_row,
    _datetime,
    _dictionary,
    _first_row,
    _review_decision,
    _review_status,
    _review_type,
)
from services.human_review import CreateHumanReview, HumanReviewCase

CASE_ID = "11111111-1111-4111-8111-111111111111"
CONVERSATION_ID = "22222222-2222-4222-8222-222222222222"
RUN_ID = "33333333-3333-4333-8333-333333333333"


def make_query(data: object) -> MagicMock:
    query = MagicMock()
    for method in ("select", "eq", "limit", "order"):
        getattr(query, method).return_value = query
    query.execute.return_value = SimpleNamespace(data=data)
    return query


def sample_case_row(
    status: str = "pending",
    claimed_by: str | None = None,
    decision: str | None = None,
) -> dict[str, object]:
    now_iso = "2026-11-11T12:00:00+00:00"
    return {
        "case_id": CASE_ID,
        "tenant_id": "tenant-1",
        "conversation_id": CONVERSATION_ID,
        "run_id": RUN_ID,
        "review_type": "clinical",
        "status": status,
        "requested_action": "Review dosage",
        "context": {"query": "Review dosage"},
        "claimed_by": claimed_by,
        "claimed_at": now_iso if claimed_by else None,
        "decision": decision,
        "resolution": {"approved": True} if decision else None,
        "resolved_by": "staff-1" if decision else None,
        "resolved_at": now_iso if decision else None,
        "expires_at": now_iso,
        "created_at": now_iso,
        "updated_at": now_iso,
    }


@pytest.mark.anyio
async def test_expire_cases_calls_rpc() -> None:
    client = MagicMock()
    rpc_query = MagicMock()
    rpc_query.execute.return_value = SimpleNamespace(data=3)
    client.rpc.return_value = rpc_query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.expire_cases("tenant-1")

    assert result == 3
    client.rpc.assert_called_once_with(
        "expire_human_review_cases",
        {"p_tenant_id": "tenant-1"},
    )


@pytest.mark.anyio
async def test_create_case_calls_rpc() -> None:
    client = MagicMock()
    rpc_query = MagicMock()
    rpc_query.execute.return_value = SimpleNamespace(data=[sample_case_row()])
    client.rpc.return_value = rpc_query

    store = SupabaseHumanReviewStore(client=client)
    now = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    command = CreateHumanReview(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        conversation_id=CONVERSATION_ID,
        run_id=RUN_ID,
        review_type="clinical",
        requested_action="Review dosage",
        context={"query": "Review dosage"},
        expires_at=now,
        created_at=now,
    )

    result = await store.create_case(command)

    assert isinstance(result, HumanReviewCase)
    assert result.case_id == CASE_ID
    client.rpc.assert_called_once_with(
        "create_human_review_case",
        {
            "p_case_id": CASE_ID,
            "p_tenant_id": "tenant-1",
            "p_conversation_id": CONVERSATION_ID,
            "p_run_id": RUN_ID,
            "p_review_type": "clinical",
            "p_requested_action": "Review dosage",
            "p_context": {"query": "Review dosage"},
            "p_expires_at": now.isoformat(),
            "p_created_at": now.isoformat(),
        },
    )


@pytest.mark.anyio
async def test_get_case_returns_case_when_found() -> None:
    client = MagicMock()
    query = make_query([sample_case_row()])
    client.table.return_value = query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.get_case(case_id=CASE_ID, tenant_id="tenant-1")

    assert isinstance(result, HumanReviewCase)
    assert result.case_id == CASE_ID
    client.table.assert_called_once_with("human_review_cases")


@pytest.mark.anyio
async def test_get_case_returns_none_when_empty() -> None:
    client = MagicMock()
    query = make_query([])
    client.table.return_value = query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.get_case(case_id=CASE_ID, tenant_id="tenant-1")

    assert result is None


@pytest.mark.anyio
async def test_list_cases_with_status_filter() -> None:
    client = MagicMock()
    query = make_query([sample_case_row()])
    client.table.return_value = query

    store = SupabaseHumanReviewStore(client=client)
    results = await store.list_cases(
        tenant_id="tenant-1",
        status="pending",
        limit=20,
    )

    assert len(results) == 1
    assert results[0].case_id == CASE_ID
    query.eq.assert_any_call("tenant_id", "tenant-1")
    query.eq.assert_any_call("status", "pending")
    query.limit.assert_called_once_with(20)


@pytest.mark.anyio
async def test_list_cases_without_status_filter() -> None:
    client = MagicMock()
    query = make_query([sample_case_row()])
    client.table.return_value = query

    store = SupabaseHumanReviewStore(client=client)
    results = await store.list_cases(
        tenant_id="tenant-1",
        status=None,
        limit=50,
    )

    assert len(results) == 1
    query.limit.assert_called_once_with(50)


@pytest.mark.anyio
async def test_claim_case_calls_rpc() -> None:
    client = MagicMock()
    rpc_query = MagicMock()
    rpc_query.execute.return_value = SimpleNamespace(
        data=[sample_case_row(status="claimed", claimed_by="staff-1")]
    )
    client.rpc.return_value = rpc_query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.claim_case(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
    )

    assert isinstance(result, HumanReviewCase)
    assert result.status == "claimed"
    assert result.claimed_by == "staff-1"
    client.rpc.assert_called_once_with(
        "claim_human_review_case",
        {
            "p_case_id": CASE_ID,
            "p_tenant_id": "tenant-1",
            "p_staff_id": "staff-1",
        },
    )


@pytest.mark.anyio
async def test_claim_case_returns_none_when_empty() -> None:
    client = MagicMock()
    rpc_query = MagicMock()
    rpc_query.execute.return_value = SimpleNamespace(data=[])
    client.rpc.return_value = rpc_query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.claim_case(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
    )

    assert result is None


@pytest.mark.anyio
async def test_resolve_case_calls_rpc() -> None:
    client = MagicMock()
    rpc_query = MagicMock()
    rpc_query.execute.return_value = SimpleNamespace(
        data=[sample_case_row(status="resolved", decision="approved")]
    )
    client.rpc.return_value = rpc_query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.resolve_case(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={"verified": True},
    )

    assert isinstance(result, HumanReviewCase)
    assert result.status == "resolved"
    assert result.decision == "approved"
    client.rpc.assert_called_once_with(
        "resolve_human_review_case",
        {
            "p_case_id": CASE_ID,
            "p_tenant_id": "tenant-1",
            "p_staff_id": "staff-1",
            "p_decision": "approved",
            "p_resolution": {"verified": True},
        },
    )


@pytest.mark.anyio
async def test_resolve_case_returns_none_when_empty() -> None:
    client = MagicMock()
    rpc_query = MagicMock()
    rpc_query.execute.return_value = SimpleNamespace(data=[])
    client.rpc.return_value = rpc_query

    store = SupabaseHumanReviewStore(client=client)
    result = await store.resolve_case(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={},
    )

    assert result is None


def test_parsers_and_validation_errors() -> None:
    with pytest.raises(RuntimeError, match="invalid review response"):
        _first_row([])

    with pytest.raises(RuntimeError, match="invalid review response"):
        _first_row("not-a-dict")

    assert _first_row({"case_id": "1"}) == {"case_id": "1"}

    with pytest.raises(RuntimeError, match="invalid review context"):
        _dictionary("not-a-dict")

    with pytest.raises(RuntimeError, match="invalid timestamp"):
        _datetime(12345)

    with pytest.raises(RuntimeError, match="invalid review type"):
        _review_type("unsupported_type")

    with pytest.raises(RuntimeError, match="invalid review status"):
        _review_status("unsupported_status")

    with pytest.raises(RuntimeError, match="invalid review decision"):
        _review_decision("unsupported_decision")

    assert _review_decision(None) is None
    assert _review_decision("approved") == "approved"


def test_case_from_row_mapping() -> None:
    row = sample_case_row()
    case = _case_from_row(row)
    assert case.case_id == CASE_ID
    assert case.tenant_id == "tenant-1"
    assert case.review_type == "clinical"
    assert case.status == "pending"
    assert case.decision is None
