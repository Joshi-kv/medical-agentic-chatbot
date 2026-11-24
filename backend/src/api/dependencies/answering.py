from functools import lru_cache

from api.dependencies.evidence import get_evidence_service
from api.dependencies.web_search import get_optional_web_search_workflow
from integrations.nvidia_nim import get_nvidia_nim_chat_model
from services.answering import AnsweringService


@lru_cache
def get_answering_service() -> AnsweringService:
    return AnsweringService(
        evidence_service=get_evidence_service(),
        web_search_workflow=get_optional_web_search_workflow(),
        chat_model=get_nvidia_nim_chat_model(),
    )
