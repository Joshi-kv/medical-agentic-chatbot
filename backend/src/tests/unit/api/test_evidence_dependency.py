from unittest.mock import MagicMock, patch

from api.dependencies.evidence import get_evidence_service
from services.evidence import EvidenceService
from services.retrieval import RetrievalService


def test_get_evidence_service_initialization() -> None:
    mock_retrieval_service = MagicMock(spec=RetrievalService)

    with patch(
        "api.dependencies.evidence.get_retrieval_service",
        return_value=mock_retrieval_service,
    ):
        get_evidence_service.cache_clear()
        service = get_evidence_service()

        assert isinstance(service, EvidenceService)
        assert service._retrieval_service is mock_retrieval_service


def test_get_evidence_service_caches_instance() -> None:
    mock_retrieval_service = MagicMock(spec=RetrievalService)

    with patch(
        "api.dependencies.evidence.get_retrieval_service",
        return_value=mock_retrieval_service,
    ):
        get_evidence_service.cache_clear()
        service_1 = get_evidence_service()
        service_2 = get_evidence_service()

        assert service_1 is service_2
