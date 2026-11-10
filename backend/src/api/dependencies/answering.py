from functools import lru_cache

from api.dependencies.evidence import get_evidence_service
from integrations.nvidia_nim import get_nvidia_nim_chat_model
from services.answering import AnsweringService


@lru_cache
def get_answering_service() -> AnsweringService:
    """Return the AnsweringService."""
    return AnsweringService(
        evidence_service=get_evidence_service(),
        chat_model=get_nvidia_nim_chat_model(),
    )
