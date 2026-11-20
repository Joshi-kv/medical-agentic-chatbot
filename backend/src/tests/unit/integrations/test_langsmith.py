import os
from types import SimpleNamespace

import pytest
from langsmith.utils import get_tracer_project, tracing_is_enabled
from pydantic import SecretStr

from integrations.langsmith import configure_langsmith_tracing


def make_settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "langsmith_tracing": True,
        "langsmith_api_key": SecretStr("unit-test-placeholder"),
        "langsmith_project": "medical-agentic-chatbot-test",
        "langsmith_endpoint": "https://api.smith.langchain.com",
        "langsmith_hide_inputs": True,
        "langsmith_hide_outputs": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def clear_langsmith_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "LANGSMITH_HIDE_INPUTS",
        "LANGSMITH_HIDE_OUTPUTS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_configure_langsmith_tracing_sets_safe_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_langsmith_environment(monkeypatch)

    enabled = configure_langsmith_tracing(make_settings())  # type: ignore[arg-type]

    assert enabled is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "unit-test-placeholder"
    assert os.environ["LANGSMITH_PROJECT"] == "medical-agentic-chatbot-test"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert tracing_is_enabled() is True
    assert get_tracer_project() == "medical-agentic-chatbot-test"


def test_configure_langsmith_tracing_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_langsmith_environment(monkeypatch)

    enabled = configure_langsmith_tracing(  # type: ignore[arg-type]
        make_settings(
            langsmith_tracing=False,
            langsmith_api_key=None,
        )
    )

    assert enabled is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_configure_langsmith_tracing_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_langsmith_environment(monkeypatch)

    with pytest.raises(ValueError, match="LANGSMITH_API_KEY"):
        configure_langsmith_tracing(  # type: ignore[arg-type]
            make_settings(langsmith_api_key=None)
        )
