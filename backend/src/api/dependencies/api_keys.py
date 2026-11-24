from functools import lru_cache

from integrations.supabase import get_supabase_client
from repositories.supabase_api_keys import SupabaseAPIKeyStore
from services.api_keys import APIKeyService


@lru_cache
def get_api_key_service() -> APIKeyService:
    return APIKeyService(SupabaseAPIKeyStore(get_supabase_client()))
