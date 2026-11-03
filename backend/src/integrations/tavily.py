from functools import lru_cache

from tavily import TavilyClient

from config.settings import get_settings


@lru_cache()
def get_tavily_client() -> TavilyClient:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise ValueError("Tavily API key not configured.")
    return TavilyClient(api_key=settings.tavily_api_key)
