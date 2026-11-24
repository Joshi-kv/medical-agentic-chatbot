from types import SimpleNamespace
from unittest.mock import patch

from api.dependencies.web_search import get_optional_web_search_workflow


def test_optional_web_search_is_disabled_without_api_key() -> None:
    settings = SimpleNamespace(tavily_api_key=None)

    with patch("api.dependencies.web_search.get_settings", return_value=settings):
        assert get_optional_web_search_workflow() is None
