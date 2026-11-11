from functools import lru_cache

from api.dependencies.answering import get_answering_service
from guardrails.emergency import DEFAULT_EMERGENCY_RULES, EmergencyGuardrail
from integrations.nvidia_nim import get_nvidia_nim_chat_model
from services.intent_router import IntentRouter
from services.medical_agent import MedicalAgent


@lru_cache
def get_medical_agent() -> MedicalAgent:
    return MedicalAgent(
        emergency_guardrail=EmergencyGuardrail(rules=DEFAULT_EMERGENCY_RULES),
        intent_router=IntentRouter(chat_model=get_nvidia_nim_chat_model()),
        answering_service=get_answering_service(),
    )
