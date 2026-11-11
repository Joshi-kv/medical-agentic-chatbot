from functools import lru_cache

from api.dependencies.medical_agent import get_medical_agent
from config.settings import get_settings
from integrations.supabase import get_supabase_client
from repositories.supabase_conversations import SupabaseConversationStore
from services.conversation import ConversationService
from services.turns import TurnService


@lru_cache
def get_conversation_store() -> SupabaseConversationStore:
    return SupabaseConversationStore(client=get_supabase_client())


@lru_cache
def get_conversation_service() -> ConversationService:
    return ConversationService(store=get_conversation_store())


@lru_cache
def get_turn_service() -> TurnService:
    settings = get_settings()
    if not settings.nvidia_chat_model:
        raise ValueError("NVIDIA chat model is not configured.")

    return TurnService(
        conversations=get_conversation_service(),
        store=get_conversation_store(),
        answering=get_medical_agent(),
        model_name=settings.nvidia_chat_model,
    )
