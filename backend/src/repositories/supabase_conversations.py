from datetime import datetime
from typing import Any, cast
from uuid import uuid4

from anyio import to_thread
from supabase import Client

from services.conversation import (
    CompleteTurnCommand,
    Conversation,
    ConversationStatus,
    Message,
    MessageRole,
    StartedTurn,
)


class SupabaseConversationStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    async def create_conversation(
        self,
        conversation: Conversation,
    ) -> Conversation:
        response = await to_thread.run_sync(
            lambda: self._client.table("conversations")
            .insert(
                {
                    "conversation_id": conversation.conversation_id,
                    "tenant_id": conversation.tenant_id,
                    "actor_id": conversation.actor_id,
                    "title": conversation.title,
                    "summary": conversation.summary,
                    "status": conversation.status,
                    "created_at": conversation.created_at.isoformat(),
                    "updated_at": conversation.updated_at.isoformat(),
                }
            )
            .execute()
        )
        return _conversation_from_row(_first_row(response.data))

    async def get_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
    ) -> Conversation | None:
        response = await to_thread.run_sync(
            lambda: self._client.table("conversations")
            .select(
                "conversation_id,tenant_id,actor_id,title,summary,"
                "status,created_at,updated_at"
            )
            .eq("conversation_id", conversation_id)
            .eq("tenant_id", tenant_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return _conversation_from_row(_first_row(response.data))

    async def list_messages(
        self,
        conversation_id: str,
        tenant_id: str,
    ) -> list[Message]:
        response = await to_thread.run_sync(
            lambda: self._client.table("messages")
            .select(
                "message_id,conversation_id,tenant_id,role,content,created_at"
            )
            .eq("conversation_id", conversation_id)
            .eq("tenant_id", tenant_id)
            .order("created_at")
            .execute()
        )
        return [_message_from_row(row) for row in response.data]

    async def start_turn(self, message: Message) -> StartedTurn:
        run_id = _new_uuid()
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "start_conversation_turn",
                {
                    "p_run_id": run_id,
                    "p_user_message_id": message.message_id,
                    "p_conversation_id": message.conversation_id,
                    "p_tenant_id": message.tenant_id,
                    "p_content": message.content,
                    "p_created_at": message.created_at.isoformat(),
                },
            ).execute()
        )
        row = _first_row(response.data)
        return StartedTurn(
            run_id=str(row["run_id"]),
            user_message=_message_from_row(row),
        )

    async def complete_turn(self, command: CompleteTurnCommand) -> None:
        citations = [
            {
                "citation_id": citation.citation_id,
                "citation_label": citation.citation_label,
                "document_id": citation.document_id,
                "chunk_id": citation.chunk_id,
                "page_number": citation.page_number,
                "source": citation.source,
                "title": citation.title,
            }
            for citation in command.citations
        ]
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "complete_conversation_turn",
                {
                    "p_run_id": command.run_id,
                    "p_tenant_id": command.tenant_id,
                    "p_assistant_message_id": command.assistant_message.message_id,
                    "p_content": command.assistant_message.content,
                    "p_citations": citations,
                    "p_model": command.model,
                    "p_route": command.route,
                    "p_latency_ms": command.latency_ms,
                    "p_created_at": command.assistant_message.created_at.isoformat(),
                },
            ).execute()
        )
        if not _scalar_boolean(response.data):
            raise RuntimeError("Supabase did not complete the conversation turn.")

    async def fail_turn(
        self,
        run_id: str,
        tenant_id: str,
        error_code: str,
    ) -> None:
        response = await to_thread.run_sync(
            lambda: self._client.rpc(
                "fail_conversation_turn",
                {
                    "p_run_id": run_id,
                    "p_tenant_id": tenant_id,
                    "p_error_code": error_code,
                },
            ).execute()
        )
        if not _scalar_boolean(response.data):
            raise RuntimeError("Supabase did not fail the conversation turn.")


def _conversation_from_row(row: dict[str, Any]) -> Conversation:
    return Conversation(
        conversation_id=str(row["conversation_id"]),
        tenant_id=str(row["tenant_id"]),
        actor_id=str(row["actor_id"]),
        title=_optional_string(row.get("title")),
        summary=_optional_string(row.get("summary")),
        status=_conversation_status(row["status"]),
        created_at=_parse_datetime(row["created_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _message_from_row(row: dict[str, Any]) -> Message:
    return Message(
        message_id=str(row["message_id"]),
        conversation_id=str(row["conversation_id"]),
        tenant_id=str(row["tenant_id"]),
        role=_message_role(row["role"]),
        content=str(row["content"]),
        created_at=_parse_datetime(row["created_at"]),
    )


def _first_row(data: object) -> dict[str, Any]:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    raise RuntimeError("Supabase returned an unexpected conversation response.")


def _scalar_boolean(data: object) -> bool:
    if isinstance(data, list):
        return bool(data[0]) if data else False
    return bool(data)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _conversation_status(value: object) -> ConversationStatus:
    status = str(value)
    if status not in {"active", "archived", "deleted"}:
        raise RuntimeError("Supabase returned an invalid conversation status.")
    return cast(ConversationStatus, status)


def _message_role(value: object) -> MessageRole:
    role = str(value)
    if role not in {"user", "assistant", "system"}:
        raise RuntimeError("Supabase returned an invalid message role.")
    return cast(MessageRole, role)


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError("Supabase returned an invalid timestamp.")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _new_uuid() -> str:
    return str(uuid4())
