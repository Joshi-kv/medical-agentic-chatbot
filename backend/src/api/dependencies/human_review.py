from functools import lru_cache

from integrations.supabase import get_supabase_client
from repositories.supabase_human_review import (
    SupabaseHumanReviewStore,
)
from services.human_review import HumanReviewService


@lru_cache
def get_human_review_store() -> SupabaseHumanReviewStore:
    return SupabaseHumanReviewStore(
        client=get_supabase_client(),
    )


@lru_cache
def get_human_review_service() -> HumanReviewService:
    return HumanReviewService(
        store=get_human_review_store(),
    )
