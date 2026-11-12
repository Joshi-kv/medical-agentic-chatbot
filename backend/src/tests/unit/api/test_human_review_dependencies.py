from unittest.mock import MagicMock, patch

from api.dependencies.human_review import (
    get_human_review_service,
    get_human_review_store,
)
from repositories.supabase_human_review import SupabaseHumanReviewStore
from services.human_review import HumanReviewService


def clear_caches() -> None:
    get_human_review_service.cache_clear()
    get_human_review_store.cache_clear()


def test_get_human_review_dependencies() -> None:
    client = MagicMock()
    with patch(
        "api.dependencies.human_review.get_supabase_client",
        return_value=client,
    ):
        clear_caches()
        store = get_human_review_store()
        service = get_human_review_service()

        assert isinstance(store, SupabaseHumanReviewStore)
        assert isinstance(service, HumanReviewService)
        assert service._store is store

        # Verify caching
        assert get_human_review_store() is store
        assert get_human_review_service() is service
    clear_caches()
