from functools import lru_cache

from tavily import TavilyClient

from config.settings import get_settings


@lru_cache()
def get_tavily_client() -> TavilyClient:
    settings = get_settings()
    if settings.tavily_api_key is None:
        raise ValueError("Tavily API key not configured.")

    api_key = settings.tavily_api_key.get_secret_value()
    if not api_key:
        raise ValueError("Tavily API key not configured.")

    return TavilyClient(api_key=api_key)
