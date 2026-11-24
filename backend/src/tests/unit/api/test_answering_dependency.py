from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel

from api.dependencies.answering import get_answering_service
from services.answering import AnsweringService
from services.evidence import EvidenceService
from services.web_search_workflow import WebSearchWorkflow


def test_get_answering_service_initialization() -> None:
    mock_evidence_service = MagicMock(spec=EvidenceService)
    mock_chat_model = MagicMock(spec=BaseChatModel)
    mock_web_search = MagicMock(spec=WebSearchWorkflow)

    with (
        patch(
            "api.dependencies.answering.get_evidence_service",
            return_value=mock_evidence_service,
        ),
        patch(
            "api.dependencies.answering.get_nvidia_nim_chat_model",
            return_value=mock_chat_model,
        ),
        patch(
            "api.dependencies.answering.get_optional_web_search_workflow",
            return_value=mock_web_search,
        ),
    ):
        get_answering_service.cache_clear()
        service = get_answering_service()

        assert isinstance(service, AnsweringService)
        assert service._evidence_service is mock_evidence_service
        assert service._chat_model is mock_chat_model
        assert service._web_search_workflow is mock_web_search


def test_get_answering_service_caches_instance() -> None:
    mock_evidence_service = MagicMock(spec=EvidenceService)
    mock_chat_model = MagicMock(spec=BaseChatModel)
    mock_web_search = MagicMock(spec=WebSearchWorkflow)

    with (
        patch(
            "api.dependencies.answering.get_evidence_service",
            return_value=mock_evidence_service,
        ),
        patch(
            "api.dependencies.answering.get_nvidia_nim_chat_model",
            return_value=mock_chat_model,
        ),
        patch(
            "api.dependencies.answering.get_optional_web_search_workflow",
            return_value=mock_web_search,
        ),
    ):
        get_answering_service.cache_clear()
        service_1 = get_answering_service()
        service_2 = get_answering_service()

        assert service_1 is service_2
