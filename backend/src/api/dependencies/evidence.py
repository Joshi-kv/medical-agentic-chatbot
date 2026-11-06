from functools import lru_cache

from api.dependencies.retrieval import get_retrieval_service
from services.evidence import EvidenceService


@lru_cache
def get_evidence_service() -> EvidenceService:
    """Return the EvidenceService."""
    return EvidenceService(retrieval_service=get_retrieval_service())
