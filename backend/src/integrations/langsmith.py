import os

from langsmith import utils as langsmith_utils

from config.settings import Settings


def configure_langsmith_tracing(settings: Settings) -> bool:
    os.environ["LANGSMITH_TRACING"] = str(settings.langsmith_tracing).lower()

    if not settings.langsmith_tracing:
        _clear_langsmith_environment_cache()
        return False

    if settings.langsmith_api_key is None:
        raise ValueError("LANGSMITH_API_KEY is required when tracing is enabled.")

    api_key = settings.langsmith_api_key.get_secret_value().strip()
    project = settings.langsmith_project.strip()
    endpoint = settings.langsmith_endpoint.strip()

    if not api_key:
        raise ValueError("LANGSMITH_API_KEY is required when tracing is enabled.")

    if not project:
        raise ValueError("LANGSMITH_PROJECT is required when tracing is enabled.")

    if not endpoint:
        raise ValueError("LANGSMITH_ENDPOINT is required when tracing is enabled.")

    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = project
    os.environ["LANGSMITH_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_HIDE_INPUTS"] = str(settings.langsmith_hide_inputs).lower()
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(settings.langsmith_hide_outputs).lower()

    _clear_langsmith_environment_cache()

    return True


def _clear_langsmith_environment_cache() -> None:
    langsmith_utils.get_env_var.cache_clear()
    langsmith_utils.get_tracer_project.cache_clear()
