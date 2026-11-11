from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from api.dependencies.conversations import (
    get_conversation_service,
    get_conversation_store,
    get_turn_service,
)
from repositories.supabase_conversations import SupabaseConversationStore
from services.conversation import ConversationService
from services.turns import TurnService


def clear_caches() -> None:
    get_turn_service.cache_clear()
    get_conversation_service.cache_clear()
    get_conversation_store.cache_clear()


def test_get_conversation_service_reuses_store() -> None:
    client = MagicMock()
    with patch(
        "api.dependencies.conversations.get_supabase_client",
        return_value=client,
    ):
        clear_caches()
        store = get_conversation_store()
        service = get_conversation_service()

        assert isinstance(store, SupabaseConversationStore)
        assert isinstance(service, ConversationService)
        assert service._store is store
    clear_caches()


def test_get_turn_service_wires_dependencies() -> None:
    conversation_service = MagicMock(spec=ConversationService)
    store = MagicMock(spec=SupabaseConversationStore)
    answering = MagicMock()
    settings = SimpleNamespace(nvidia_chat_model="nvidia/test-model")

    with (
        patch(
            "api.dependencies.conversations.get_conversation_service",
            return_value=conversation_service,
        ),
        patch(
            "api.dependencies.conversations.get_conversation_store",
            return_value=store,
        ),
        patch(
            "api.dependencies.conversations.get_answering_service",
            return_value=answering,
        ),
        patch(
            "api.dependencies.conversations.get_settings",
            return_value=settings,
        ),
    ):
        clear_caches()
        service = get_turn_service()

        assert isinstance(service, TurnService)
        assert service._conversations is conversation_service
        assert service._store is store
        assert service._answering is answering
        assert service._model_name == "nvidia/test-model"
    clear_caches()
