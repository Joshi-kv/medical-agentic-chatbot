from datetime import datetime, timezone
from unittest.mock import AsyncMock

from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.dependencies.conversations import (
    get_conversation_service,
    get_turn_service,
)
from api.routers.conversations import router
from api.security.api_key import APIPrincipal, authenticate_api_key
from services.conversation import (
    CitationRecord,
    Conversation,
    ConversationNotFound,
    ConversationService,
    Message,
)
from services.turns import TurnResult, TurnService

CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"


def make_conversation() -> Conversation:
    now = datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)
    return Conversation(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        actor_id="key-1",
        title="Heart health",
        summary=None,
        status="active",
        created_at=now,
        updated_at=now,
    )


def make_client(
    conversation_service: object,
    turn_service: object,
    scopes: frozenset[str] | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_conversation_service] = (
        lambda: conversation_service
    )
    app.dependency_overrides[get_turn_service] = lambda: turn_service
    principal = APIPrincipal(
        key_id="key-1",
        tenant_id="tenant-1",
        scopes=scopes
        or frozenset({"conversations:read", "conversations:write"}),
    )
    app.dependency_overrides[authenticate_api_key] = lambda: principal
    return TestClient(app, raise_server_exceptions=False)


def test_create_conversation_uses_authenticated_identity() -> None:
    service = AsyncMock(spec=ConversationService)
    service.create.return_value = make_conversation()
    client = make_client(service, AsyncMock(spec=TurnService))

    response = client.post(
        "/v1/conversations",
        json={"title": "Heart health"},
    )

    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["conversation_id"] == CONVERSATION_ID
    service.create.assert_awaited_once_with(
        tenant_id="tenant-1",
        actor_id="key-1",
        title="Heart health",
    )


def test_get_conversation_uses_authenticated_tenant() -> None:
    service = AsyncMock(spec=ConversationService)
    service.get.return_value = make_conversation()
    client = make_client(service, AsyncMock(spec=TurnService))

    response = client.get(f"/v1/conversations/{CONVERSATION_ID}")

    assert response.status_code == status.HTTP_200_OK
    service.get.assert_awaited_once_with(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
    )


def test_get_conversation_returns_404_for_missing_record() -> None:
    service = AsyncMock(spec=ConversationService)
    service.get.side_effect = ConversationNotFound
    client = make_client(service, AsyncMock(spec=TurnService))

    response = client.get(f"/v1/conversations/{CONVERSATION_ID}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"]["code"] == "conversation_not_found"


def test_list_messages_returns_chronological_records() -> None:
    service = AsyncMock(spec=ConversationService)
    message = Message(
        message_id="22222222-2222-4222-8222-222222222222",
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        role="user",
        content="What is hypertension?",
        created_at=datetime(2026, 11, 10, 10, 1, tzinfo=timezone.utc),
    )
    service.list_messages.return_value = [message]
    client = make_client(service, AsyncMock(spec=TurnService))

    response = client.get(
        f"/v1/conversations/{CONVERSATION_ID}/messages"
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()[0]["content"] == "What is hypertension?"
    service.list_messages.assert_awaited_once_with(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
    )


def test_create_turn_returns_persisted_answer() -> None:
    conversation_service = AsyncMock(spec=ConversationService)
    turn_service = AsyncMock(spec=TurnService)
    assistant_message = Message(
        message_id="22222222-2222-4222-8222-222222222222",
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        role="assistant",
        content="Hypertension is elevated blood pressure [C1].",
        created_at=datetime(2026, 11, 10, 10, 2, tzinfo=timezone.utc),
    )
    citation = CitationRecord(
        citation_id="33333333-3333-4333-8333-333333333333",
        tenant_id="tenant-1",
        assistant_message_id=assistant_message.message_id,
        citation_label="C1",
        document_id="document-1",
        chunk_id="chunk-1",
        page_number=12,
        source="book.pdf",
        title="Medical Book",
    )
    turn_service.run.return_value = TurnResult(
        run_id="44444444-4444-4444-8444-444444444444",
        message=assistant_message,
        citations=[citation],
        sufficient_evidence=True,
        safety_notice="Educational only.",
    )
    client = make_client(conversation_service, turn_service)

    response = client.post(
        f"/v1/conversations/{CONVERSATION_ID}/turns",
        json={"query": "What is hypertension?"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["citations"][0]["citation_label"] == "C1"
    turn_service.run.assert_awaited_once_with(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        query="What is hypertension?",
        document_id=None,
    )


def test_conversation_route_requires_correct_scope() -> None:
    service = AsyncMock(spec=ConversationService)
    client = make_client(
        service,
        AsyncMock(spec=TurnService),
        scopes=frozenset({"documents:read"}),
    )

    response = client.get(f"/v1/conversations/{CONVERSATION_ID}")

    assert response.status_code == status.HTTP_403_FORBIDDEN
    service.get.assert_not_awaited()


def test_invalid_conversation_uuid_returns_422() -> None:
    service = AsyncMock(spec=ConversationService)
    client = make_client(service, AsyncMock(spec=TurnService))

    response = client.get("/v1/conversations/not-a-uuid")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    service.get.assert_not_awaited()
