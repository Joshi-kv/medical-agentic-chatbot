from datetime import datetime
from typing import Any, cast

from anyio import to_thread
from supabase import Client

from services.human_review import (
    CreateHumanReview,
    HumanReviewCase,
    ReviewDecision,
    ReviewStatus,
    ReviewType,
)

CASE_COLUMNS = (
    "case_id,tenant_id,conversation_id,run_id,review_type,status,"
    "requested_action,context,claimed_by,claimed_at,decision,resolution,"
    "resolved_by,resolved_at,expires_at,created_at,updated_at"
)


class SupabaseHumanReviewStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def expire_cases(self, tenant_id: str) -> int:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "expire_human_review_cases",
                {"p_tenant_id": tenant_id},
            ).execute()
        )

        data = response.data
        if isinstance(data, list):
            return int(data[0]) if data else 0
        return int(data or 0)

    async def create_case(
        self,
        command: CreateHumanReview,
    ) -> HumanReviewCase:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "create_human_review_case",
                {
                    "p_case_id": command.case_id,
                    "p_tenant_id": command.tenant_id,
                    "p_conversation_id": command.conversation_id,
                    "p_run_id": command.run_id,
                    "p_review_type": command.review_type,
                    "p_requested_action": command.requested_action,
                    "p_context": command.context,
                    "p_expires_at": command.expires_at.isoformat(),
                    "p_created_at": command.created_at.isoformat(),
                },
            ).execute()
        )

        return _case_from_row(_first_row(response.data))

    async def get_case(
        self,
        case_id: str,
        tenant_id: str,
    ) -> HumanReviewCase | None:
        response = await to_thread.run_sync(
            lambda: (
                self._client.table("human_review_cases")
                .select(CASE_COLUMNS)
                .eq("case_id", case_id)
                .eq("tenant_id", tenant_id)
                .limit(1)
                .execute()
            )
        )

        if not response.data:
            return None

        return _case_from_row(_first_row(response.data))

    async def list_cases(
        self,
        tenant_id: str,
        status: ReviewStatus | None,
        limit: int,
    ) -> list[HumanReviewCase]:
        def execute_query():
            query = (
                self._client.table("human_review_cases")
                .select(CASE_COLUMNS)
                .eq("tenant_id", tenant_id)
            )

            if status is not None:
                query = query.eq("status", status)

            return query.order("created_at", desc=False).limit(limit).execute()

        response = await to_thread.run_sync(execute_query)

        return [_case_from_row(row) for row in response.data]

    async def claim_case(
        self,
        case_id: str,
        tenant_id: str,
        staff_id: str,
    ) -> HumanReviewCase | None:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "claim_human_review_case",
                {
                    "p_case_id": case_id,
                    "p_tenant_id": tenant_id,
                    "p_staff_id": staff_id,
                },
            ).execute()
        )

        if not response.data:
            return None

        return _case_from_row(_first_row(response.data))

    async def resolve_case(
        self,
        case_id: str,
        tenant_id: str,
        staff_id: str,
        decision: ReviewDecision,
        resolution: dict[str, Any],
    ) -> HumanReviewCase | None:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "resolve_human_review_case",
                {
                    "p_case_id": case_id,
                    "p_tenant_id": tenant_id,
                    "p_staff_id": staff_id,
                    "p_decision": decision,
                    "p_resolution": resolution,
                },
            ).execute()
        )

        if not response.data:
            return None

        return _case_from_row(_first_row(response.data))


def _case_from_row(row: dict[str, Any]) -> HumanReviewCase:
    return HumanReviewCase(
        case_id=str(row["case_id"]),
        tenant_id=str(row["tenant_id"]),
        conversation_id=str(row["conversation_id"]),
        run_id=str(row["run_id"]),
        review_type=_review_type(row["review_type"]),
        status=_review_status(row["status"]),
        requested_action=str(row["requested_action"]),
        context=_dictionary(row.get("context")),
        claimed_by=_optional_string(row.get("claimed_by")),
        claimed_at=_optional_datetime(row.get("claimed_at")),
        decision=_review_decision(row.get("decision")),
        resolution=_optional_dictionary(row.get("resolution")),
        resolved_by=_optional_string(row.get("resolved_by")),
        resolved_at=_optional_datetime(row.get("resolved_at")),
        expires_at=_datetime(row["expires_at"]),
        created_at=_datetime(row["created_at"]),
        updated_at=_datetime(row["updated_at"]),
    )


def _first_row(data: object) -> dict[str, Any]:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]

    if isinstance(data, dict):
        return data

    raise RuntimeError("Supabase returned an invalid review response.")


def _dictionary(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("Supabase returned invalid review context.")

    return value


def _optional_dictionary(
    value: object,
) -> dict[str, Any] | None:
    if value is None:
        return None

    return _dictionary(value)


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError("Supabase returned an invalid timestamp.")

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None

    return _datetime(value)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _review_type(value: object) -> ReviewType:
    review_type = str(value)

    if review_type not in {"clinical", "scheduling"}:
        raise RuntimeError("Supabase returned an invalid review type.")

    return cast(ReviewType, review_type)


def _review_status(value: object) -> ReviewStatus:
    status = str(value)

    if status not in {
        "pending",
        "claimed",
        "resolved",
        "rejected",
        "expired",
    }:
        raise RuntimeError("Supabase returned an invalid review status.")

    return cast(ReviewStatus, status)


def _review_decision(value: object) -> ReviewDecision | None:
    if value is None:
        return None

    decision = str(value)

    if decision not in {"approved", "rejected"}:
        raise RuntimeError("Supabase returned an invalid review decision.")

    return cast(ReviewDecision, decision)
