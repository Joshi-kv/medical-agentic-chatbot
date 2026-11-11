from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

MAX_CONVERSATION_TITLE_LENGTH = 200
ConversationStatus = Literal["active", "archived", "deleted"]
MessageRole = Literal["user", "assistant", "system"]


class ConversationError(Exception):
    pass


class ConversationNotFound(ConversationError):
    pass


class InvalidConversationInput(ConversationError):
    pass


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    tenant_id: str
    actor_id: str
    title: str | None
    summary: str | None
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Message:
    message_id: str
    conversation_id: str
    tenant_id: str
    role: MessageRole
    content: str
    created_at: datetime


@dataclass(frozen=True)
class CitationRecord:
    citation_id: str
    tenant_id: str
    assistant_message_id: str
    citation_label: str
    document_id: str
    chunk_id: str
    page_number: int
    source: str
    title: str | None


@dataclass(frozen=True)
class StartedTurn:
    run_id: str
    user_message: Message


@dataclass(frozen=True)
class CompleteTurnCommand:
    run_id: str
    tenant_id: str
    assistant_message: Message
    citations: list[CitationRecord]
    model: str
    route: str
    latency_ms: int


class ConversationStore(Protocol):
    async def create_conversation(
        self,
        conversation: Conversation,
    ) -> Conversation: ...

    async def get_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
    ) -> Conversation | None: ...

    async def list_messages(
        self,
        conversation_id: str,
        tenant_id: str,
    ) -> list[Message]: ...

    async def start_turn(
        self,
        message: Message,
    ) -> StartedTurn: ...

    async def complete_turn(
        self,
        command: CompleteTurnCommand,
    ) -> None: ...

    async def fail_turn(
        self,
        run_id: str,
        tenant_id: str,
        error_code: str,
    ) -> None: ...


class ConversationService:
    def __init__(self, store: ConversationStore) -> None:
        self._store = store

    async def create(
        self,
        tenant_id: str,
        actor_id: str,
        title: str | None = None,
    ) -> Conversation:
        normalized_tenant_id = self._required_value(tenant_id, "tenant_id")
        normalized_actor_id = self._required_value(actor_id, "actor_id")
        normalized_title = self._normalize_title(title)
        now = datetime.now(timezone.utc)

        conversation = Conversation(
            conversation_id=str(uuid4()),
            tenant_id=normalized_tenant_id,
            actor_id=normalized_actor_id,
            title=normalized_title,
            summary=None,
            status="active",
            created_at=now,
            updated_at=now,
        )

        try:
            return await self._store.create_conversation(conversation)
        except Exception as error:
            raise ConversationError("Conversation creation failed.") from error

    async def get(
        self,
        conversation_id: str,
        tenant_id: str,
    ) -> Conversation:
        normalized_conversation_id = self._required_value(
            conversation_id,
            "conversation_id",
        )
        normalized_tenant_id = self._required_value(tenant_id, "tenant_id")

        try:
            conversation = await self._store.get_conversation(
                conversation_id=normalized_conversation_id,
                tenant_id=normalized_tenant_id,
            )
        except Exception as error:
            raise ConversationError("Conversation lookup failed.") from error

        if conversation is None or conversation.status == "deleted":
            raise ConversationNotFound

        return conversation

    async def list_messages(
        self,
        conversation_id: str,
        tenant_id: str,
    ) -> list[Message]:
        conversation = await self.get(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
        )
        try:
            return await self._store.list_messages(
                conversation_id=conversation.conversation_id,
                tenant_id=conversation.tenant_id,
            )
        except Exception as error:
            raise ConversationError("Message lookup failed.") from error

    def _normalize_title(self, title: str | None) -> str | None:
        if title is None:
            return None

        normalized_title = title.strip()
        if not normalized_title:
            return None

        if len(normalized_title) > MAX_CONVERSATION_TITLE_LENGTH:
            raise InvalidConversationInput(
                f"title must be at most {MAX_CONVERSATION_TITLE_LENGTH} characters."
            )

        return normalized_title

    def _required_value(self, value: str, field_name: str) -> str:
        normalized_value = value.strip()
        if not normalized_value:
            raise InvalidConversationInput(f"{field_name} is required.")

        return normalized_value
