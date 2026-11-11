from datetime import datetime, timezone
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from services.conversation import (
    Conversation,
    ConversationError,
    ConversationNotFound,
    ConversationService,
    InvalidConversationInput,
    Message,
)


def make_conversation(
    status: Literal["active", "archived", "deleted"] = "active",
    tenant_id: str = "tenant-1",
) -> Conversation:
    now = datetime.now(timezone.utc)
    return Conversation(
        conversation_id="conversation-1",
        tenant_id=tenant_id,
        actor_id="actor-1",
        title="Heart health",
        summary=None,
        status=status,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.anyio
async def test_create_conversation_normalizes_input() -> None:
    store = AsyncMock()
    store.create_conversation.side_effect = lambda conversation: conversation
    service = ConversationService(store)

    conversation = await service.create(
        tenant_id=" tenant-1 ",
        actor_id=" actor-1 ",
        title=" Heart health ",
    )

    assert conversation.tenant_id == "tenant-1"
    assert conversation.actor_id == "actor-1"
    assert conversation.title == "Heart health"
    assert conversation.status == "active"
    assert conversation.created_at.tzinfo is timezone.utc
    assert conversation.updated_at == conversation.created_at
    store.create_conversation.assert_awaited_once_with(conversation)


@pytest.mark.anyio
@pytest.mark.parametrize("title", [None, "", "   "])
async def test_create_conversation_accepts_empty_title(title: str | None) -> None:
    store = AsyncMock()
    store.create_conversation.side_effect = lambda conversation: conversation
    service = ConversationService(store)

    conversation = await service.create("tenant-1", "actor-1", title)

    assert conversation.title is None


@pytest.mark.anyio
async def test_create_conversation_rejects_long_title() -> None:
    store = AsyncMock()
    service = ConversationService(store)

    with pytest.raises(InvalidConversationInput, match="at most 200"):
        await service.create("tenant-1", "actor-1", "a" * 201)

    store.create_conversation.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tenant_id", "actor_id", "field_name"),
    [
        ("", "actor-1", "tenant_id"),
        ("tenant-1", "   ", "actor_id"),
    ],
)
async def test_create_conversation_rejects_missing_identity(
    tenant_id: str,
    actor_id: str,
    field_name: str,
) -> None:
    service = ConversationService(AsyncMock())

    with pytest.raises(InvalidConversationInput, match=field_name):
        await service.create(tenant_id, actor_id)


@pytest.mark.anyio
async def test_create_conversation_wraps_store_error() -> None:
    store = AsyncMock()
    store.create_conversation.side_effect = RuntimeError("database unavailable")
    service = ConversationService(store)

    with pytest.raises(ConversationError, match="creation failed") as exc_info:
        await service.create("tenant-1", "actor-1")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.anyio
async def test_get_conversation_uses_tenant_filter() -> None:
    expected = make_conversation()
    store = AsyncMock()
    store.get_conversation.return_value = expected
    service = ConversationService(store)

    result = await service.get("conversation-1", "tenant-1")

    assert result is expected
    store.get_conversation.assert_awaited_once_with(
        conversation_id="conversation-1",
        tenant_id="tenant-1",
    )


@pytest.mark.anyio
@pytest.mark.parametrize("conversation", [None, make_conversation(status="deleted")])
async def test_get_conversation_hides_missing_and_deleted_records(
    conversation: Conversation | None,
) -> None:
    store = AsyncMock()
    store.get_conversation.return_value = conversation
    service = ConversationService(store)

    with pytest.raises(ConversationNotFound):
        await service.get("conversation-1", "tenant-1")


@pytest.mark.anyio
async def test_get_conversation_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_conversation.side_effect = RuntimeError("database unavailable")
    service = ConversationService(store)

    with pytest.raises(ConversationError, match="lookup failed") as exc_info:
        await service.get("conversation-1", "tenant-1")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.anyio
async def test_list_messages_checks_conversation_and_returns_messages() -> None:
    conversation = make_conversation()
    message = Message(
        message_id="message-1",
        conversation_id=conversation.conversation_id,
        tenant_id=conversation.tenant_id,
        role="user",
        content="What is hypertension?",
        created_at=datetime.now(timezone.utc),
    )
    store = AsyncMock()
    store.get_conversation.return_value = conversation
    store.list_messages.return_value = [message]
    service = ConversationService(store)

    result = await service.list_messages("conversation-1", "tenant-1")

    assert result == [message]
    store.list_messages.assert_awaited_once_with(
        conversation_id="conversation-1",
        tenant_id="tenant-1",
    )


@pytest.mark.anyio
async def test_list_messages_does_not_query_messages_when_conversation_is_missing(
) -> None:
    store = AsyncMock()
    store.get_conversation.return_value = None
    service = ConversationService(store)

    with pytest.raises(ConversationNotFound):
        await service.list_messages("conversation-1", "tenant-1")

    store.list_messages.assert_not_awaited()


@pytest.mark.anyio
async def test_list_messages_wraps_store_error() -> None:
    store = AsyncMock()
    store.get_conversation.return_value = make_conversation()
    store.list_messages.side_effect = RuntimeError("database unavailable")
    service = ConversationService(store)

    with pytest.raises(ConversationError, match="Message lookup failed") as exc_info:
        await service.list_messages("conversation-1", "tenant-1")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
