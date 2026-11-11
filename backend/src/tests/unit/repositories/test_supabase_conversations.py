from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from repositories.supabase_conversations import SupabaseConversationStore
from services.conversation import (
    CitationRecord,
    CompleteTurnCommand,
    Conversation,
    Message,
)


def make_query(data: object) -> MagicMock:
    query = MagicMock()
    for method_name in ("insert", "select", "eq", "limit", "order"):
        getattr(query, method_name).return_value = query
    query.execute.return_value = SimpleNamespace(data=data)
    return query


def make_conversation() -> Conversation:
    now = datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)
    return Conversation(
        conversation_id="11111111-1111-4111-8111-111111111111",
        tenant_id="tenant-1",
        actor_id="actor-1",
        title="Heart health",
        summary=None,
        status="active",
        created_at=now,
        updated_at=now,
    )


def conversation_row() -> dict[str, object]:
    conversation = make_conversation()
    return {
        "conversation_id": conversation.conversation_id,
        "tenant_id": conversation.tenant_id,
        "actor_id": conversation.actor_id,
        "title": conversation.title,
        "summary": conversation.summary,
        "status": conversation.status,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


def make_message(role: str = "user") -> Message:
    return Message(
        message_id="22222222-2222-4222-8222-222222222222",
        conversation_id="11111111-1111-4111-8111-111111111111",
        tenant_id="tenant-1",
        role=role,
        content="What is hypertension?",
        created_at=datetime(2026, 11, 10, 10, 1, tzinfo=timezone.utc),
    )


def message_row(message: Message | None = None) -> dict[str, object]:
    message = message or make_message()
    return {
        "message_id": message.message_id,
        "conversation_id": message.conversation_id,
        "tenant_id": message.tenant_id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }


@pytest.mark.anyio
async def test_create_conversation_inserts_and_maps_row() -> None:
    query = make_query([conversation_row()])
    client = MagicMock()
    client.table.return_value = query
    store = SupabaseConversationStore(client)

    result = await store.create_conversation(make_conversation())

    assert result == make_conversation()
    client.table.assert_called_once_with("conversations")
    inserted = query.insert.call_args.args[0]
    assert inserted["tenant_id"] == "tenant-1"
    assert inserted["conversation_id"] == result.conversation_id


@pytest.mark.anyio
async def test_get_conversation_always_filters_by_tenant() -> None:
    query = make_query([conversation_row()])
    client = MagicMock()
    client.table.return_value = query
    store = SupabaseConversationStore(client)

    result = await store.get_conversation(
        "11111111-1111-4111-8111-111111111111",
        "tenant-1",
    )

    assert result == make_conversation()
    assert query.eq.call_args_list[1].args == ("tenant_id", "tenant-1")


@pytest.mark.anyio
async def test_get_conversation_returns_none_when_missing() -> None:
    query = make_query([])
    client = MagicMock()
    client.table.return_value = query
    store = SupabaseConversationStore(client)

    result = await store.get_conversation("missing", "tenant-1")

    assert result is None


@pytest.mark.anyio
async def test_list_messages_filters_and_maps_rows() -> None:
    message = make_message()
    query = make_query([message_row(message)])
    client = MagicMock()
    client.table.return_value = query
    store = SupabaseConversationStore(client)

    result = await store.list_messages(message.conversation_id, "tenant-1")

    assert result == [message]
    query.eq.assert_any_call("tenant_id", "tenant-1")
    query.order.assert_called_once_with("created_at")


@pytest.mark.anyio
async def test_start_turn_calls_transactional_rpc() -> None:
    message = make_message()
    row = {"run_id": "33333333-3333-4333-8333-333333333333", **message_row(message)}
    query = make_query([row])
    client = MagicMock()
    client.rpc.return_value = query
    store = SupabaseConversationStore(client)

    result = await store.start_turn(message)

    assert result.run_id == row["run_id"]
    assert result.user_message == message
    rpc_name, params = client.rpc.call_args.args
    assert rpc_name == "start_conversation_turn"
    assert UUID(params["p_run_id"])
    assert params["p_tenant_id"] == "tenant-1"
    assert params["p_content"] == message.content


@pytest.mark.anyio
async def test_complete_turn_sends_citations_to_rpc() -> None:
    assistant = make_message(role="assistant")
    citation = CitationRecord(
        citation_id="44444444-4444-4444-8444-444444444444",
        tenant_id="tenant-1",
        assistant_message_id=assistant.message_id,
        citation_label="C1",
        document_id="document-1",
        chunk_id="chunk-1",
        page_number=12,
        source="book.pdf",
        title="Medical Book",
    )
    command = CompleteTurnCommand(
        run_id="33333333-3333-4333-8333-333333333333",
        tenant_id="tenant-1",
        assistant_message=assistant,
        citations=[citation],
        model="chat-model",
        route="curated_documents",
        latency_ms=250,
    )
    query = make_query(True)
    client = MagicMock()
    client.rpc.return_value = query
    store = SupabaseConversationStore(client)

    await store.complete_turn(command)

    rpc_name, params = client.rpc.call_args.args
    assert rpc_name == "complete_conversation_turn"
    assert params["p_tenant_id"] == "tenant-1"
    assert params["p_citations"][0]["citation_label"] == "C1"
    assert params["p_citations"][0]["document_id"] == "document-1"


@pytest.mark.anyio
async def test_fail_turn_calls_tenant_scoped_rpc() -> None:
    query = make_query(True)
    client = MagicMock()
    client.rpc.return_value = query
    store = SupabaseConversationStore(client)

    await store.fail_turn("run-1", "tenant-1", "turn_failed")

    client.rpc.assert_called_once_with(
        "fail_conversation_turn",
        {
            "p_run_id": "run-1",
            "p_tenant_id": "tenant-1",
            "p_error_code": "turn_failed",
        },
    )
