from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

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
from services.turns import TurnError, TurnResult, TurnService, TurnStreamEvent

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
    app.dependency_overrides[get_conversation_service] = lambda: conversation_service
    app.dependency_overrides[get_turn_service] = lambda: turn_service
    principal = APIPrincipal(
        key_id="key-1",
        tenant_id="tenant-1",
        scopes=scopes or frozenset({"conversations:read", "conversations:write"}),
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

    response = client.get(f"/v1/conversations/{CONVERSATION_ID}/messages")

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
    assert response.json()["route"] == "curated_documents"
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


def test_stream_turn_returns_event_stream() -> None:
    conversation_service = AsyncMock(spec=ConversationService)
    turn_service = AsyncMock(spec=TurnService)

    async def fake_stream(**kwargs: object):
        yield TurnStreamEvent(
            event="turn_started",
            data={"run_id": "run-123"},
        )
        yield TurnStreamEvent(
            event="answer_delta",
            data={"delta": "Hypertension is high blood pressure."},
        )
        yield TurnStreamEvent(
            event="citations",
            data={"items": [{"citation_label": "C1", "source": "guide.pdf"}]},
        )
        yield TurnStreamEvent(
            event="turn_completed",
            data={"run_id": "run-123", "message_id": "msg-123"},
        )

    turn_service.stream.side_effect = fake_stream
    client = make_client(conversation_service, turn_service)

    response = client.post(
        f"/v1/conversations/{CONVERSATION_ID}/turns/stream",
        json={"query": "What is hypertension?"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert "text/event-stream" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"

    body = response.text
    assert 'event: turn_started\ndata: {"run_id": "run-123"}\n\n' in body
    assert (
        "event: answer_delta\ndata: "
        '{"delta": "Hypertension is high blood pressure."}\n\n' in body
    )
    assert "event: citations\n" in body
    assert "event: turn_completed\n" in body

    turn_service.stream.assert_called_once_with(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        query="What is hypertension?",
        document_id=None,
    )


def test_stream_turn_forwards_document_id() -> None:
    conversation_service = AsyncMock(spec=ConversationService)
    turn_service = AsyncMock(spec=TurnService)

    async def fake_stream(**kwargs: object):
        yield TurnStreamEvent(event="turn_started", data={"run_id": "run-123"})

    turn_service.stream.side_effect = fake_stream
    client = make_client(conversation_service, turn_service)
    doc_id = "55555555-5555-4555-8555-555555555555"

    response = client.post(
        f"/v1/conversations/{CONVERSATION_ID}/turns/stream",
        json={"query": "What is hypertension?", "document_id": doc_id},
    )

    assert response.status_code == status.HTTP_200_OK
    turn_service.stream.assert_called_once_with(
        conversation_id=CONVERSATION_ID,
        tenant_id="tenant-1",
        query="What is hypertension?",
        document_id=UUID(doc_id),
    )


def test_stream_turn_requires_write_scope() -> None:
    conversation_service = AsyncMock(spec=ConversationService)
    turn_service = AsyncMock(spec=TurnService)
    client = make_client(
        conversation_service,
        turn_service,
        scopes=frozenset({"conversations:read"}),
    )

    response = client.post(
        f"/v1/conversations/{CONVERSATION_ID}/turns/stream",
        json={"query": "What is hypertension?"},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    turn_service.stream.assert_not_called()


def test_stream_turn_invalid_conversation_uuid_returns_422() -> None:
    conversation_service = AsyncMock(spec=ConversationService)
    turn_service = AsyncMock(spec=TurnService)
    client = make_client(conversation_service, turn_service)

    response = client.post(
        "/v1/conversations/not-a-uuid/turns/stream",
        json={"query": "What is hypertension?"},
    )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    turn_service.stream.assert_not_called()


def test_stream_turn_formats_pre_start_failure_as_sse_error() -> None:
    conversation_service = AsyncMock(spec=ConversationService)
    turn_service = AsyncMock(spec=TurnService)

    async def failing_stream(**kwargs: object):
        if False:
            yield
        raise TurnError("database connection details")

    turn_service.stream.side_effect = failing_stream
    client = make_client(conversation_service, turn_service)

    response = client.post(
        f"/v1/conversations/{CONVERSATION_ID}/turns/stream",
        json={"query": "What is hypertension?"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert "event: error\n" in response.text
    assert '"code": "stream_unavailable"' in response.text
    assert '"discard_partial": true' in response.text
    assert "database connection details" not in response.text
