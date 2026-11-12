from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.dependencies.human_review import get_human_review_service
from api.routers.human_review import router
from api.security.api_key import (
    APIPrincipal,
    authenticate_api_key,
    authenticate_staff_api_key,
)
from services.human_review import (
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


def make_client(
    service: object,
    staff_principal: APIPrincipal | None = None,
    api_principal: APIPrincipal | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_human_review_service] = lambda: service

    default_staff = staff_principal or APIPrincipal(
        key_id="staff-1",
        tenant_id="tenant-1",
        scopes=frozenset({"human_review:read", "human_review:write"}),
    )
    app.dependency_overrides[authenticate_staff_api_key] = lambda: default_staff

    default_api = api_principal or APIPrincipal(
        key_id="key-1",
        tenant_id="tenant-1",
        scopes=frozenset({"conversations:read", "conversations:write"}),
    )
    app.dependency_overrides[authenticate_api_key] = lambda: default_api
    return TestClient(app, raise_server_exceptions=False)


def test_list_review_cases_success() -> None:
    service = AsyncMock(spec=HumanReviewService)
    case = make_case()
    service.list_cases.return_value = [case]
    client = make_client(service)

    response = client.get("/v1/human-review/cases?status=pending&limit=10")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["case_id"] == CASE_ID
    assert data[0]["status"] == "pending"
    service.list_cases.assert_awaited_once_with(
        tenant_id="tenant-1",
        status="pending",
        limit=10,
    )


def test_list_review_cases_service_error_returns_503() -> None:
    service = AsyncMock(spec=HumanReviewService)
    service.list_cases.side_effect = HumanReviewError("store error")
    client = make_client(service)

    response = client.get("/v1/human-review/cases")

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["code"] == "human_review_unavailable"


def test_get_review_case_success() -> None:
    service = AsyncMock(spec=HumanReviewService)
    case = make_case()
    service.get.return_value = case
    client = make_client(service)

    response = client.get(f"/v1/human-review/cases/{CASE_ID}")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["case_id"] == CASE_ID
    service.get.assert_awaited_once_with(case_id=CASE_ID, tenant_id="tenant-1")


def test_get_review_case_not_found_returns_404() -> None:
    service = AsyncMock(spec=HumanReviewService)
    service.get.side_effect = HumanReviewNotFound()
    client = make_client(service)

    response = client.get(f"/v1/human-review/cases/{CASE_ID}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"]["code"] == "review_case_not_found"


def test_get_review_case_invalid_uuid_returns_422() -> None:
    service = AsyncMock(spec=HumanReviewService)
    client = make_client(service)

    response = client.get("/v1/human-review/cases/not-a-uuid")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    service.get.assert_not_awaited()


def test_get_review_case_lacking_scope_returns_403() -> None:
    service = AsyncMock(spec=HumanReviewService)
    no_read_principal = APIPrincipal(
        key_id="key-1",
        tenant_id="tenant-1",
        scopes=frozenset({"conversations:write"}),
    )
    client = make_client(service, api_principal=no_read_principal)

    response = client.get(f"/v1/human-review/cases/{CASE_ID}")

    assert response.status_code == status.HTTP_403_FORBIDDEN
    service.get.assert_not_awaited()


def test_claim_review_case_success() -> None:
    service = AsyncMock(spec=HumanReviewService)
    claimed = make_case(status="claimed", claimed_by="staff-1")
    service.claim.return_value = claimed
    client = make_client(service)

    response = client.post(f"/v1/human-review/cases/{CASE_ID}/claim")

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "claimed"
    assert response.json()["claimed_by"] == "staff-1"
    service.claim.assert_awaited_once_with(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
    )


def test_claim_review_case_conflict_returns_409() -> None:
    service = AsyncMock(spec=HumanReviewService)
    service.claim.side_effect = HumanReviewConflict("Review case has expired.")
    client = make_client(service)

    response = client.post(f"/v1/human-review/cases/{CASE_ID}/claim")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["detail"]["code"] == "review_case_conflict"
    assert "expired" in response.json()["detail"]["message"]


def test_claim_review_case_not_found_returns_404() -> None:
    service = AsyncMock(spec=HumanReviewService)
    service.claim.side_effect = HumanReviewNotFound()
    client = make_client(service)

    response = client.post(f"/v1/human-review/cases/{CASE_ID}/claim")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"]["code"] == "review_case_not_found"


def test_resolve_review_case_success() -> None:
    service = AsyncMock(spec=HumanReviewService)
    resolved = make_case(status="resolved", decision="approved")
    service.resolve.return_value = resolved
    client = make_client(service)

    response = client.post(
        f"/v1/human-review/cases/{CASE_ID}/resolve",
        json={"decision": "approved", "resolution": {"note": "Looks accurate"}},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "resolved"
    assert data["decision"] == "approved"
    service.resolve.assert_awaited_once_with(
        case_id=CASE_ID,
        tenant_id="tenant-1",
        staff_id="staff-1",
        decision="approved",
        resolution={"note": "Looks accurate"},
    )


def test_resolve_review_case_conflict_returns_409() -> None:
    service = AsyncMock(spec=HumanReviewService)
    service.resolve.side_effect = HumanReviewConflict("Case already resolved.")
    client = make_client(service)

    response = client.post(
        f"/v1/human-review/cases/{CASE_ID}/resolve",
        json={"decision": "rejected", "resolution": {}},
    )

    assert response.status_code == status.HTTP_409_CONFLICT
    assert response.json()["detail"]["code"] == "review_case_conflict"


def test_resolve_review_case_invalid_input_returns_400() -> None:
    service = AsyncMock(spec=HumanReviewService)
    service.resolve.side_effect = InvalidHumanReviewInput("Invalid resolution object")
    client = make_client(service)

    response = client.post(
        f"/v1/human-review/cases/{CASE_ID}/resolve",
        json={"decision": "approved", "resolution": {}},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"]["code"] == "invalid_review_request"


def test_resolve_review_case_invalid_decision_returns_422() -> None:
    service = AsyncMock(spec=HumanReviewService)
    client = make_client(service)

    response = client.post(
        f"/v1/human-review/cases/{CASE_ID}/resolve",
        json={"decision": "undecided", "resolution": {}},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    service.resolve.assert_not_awaited()
