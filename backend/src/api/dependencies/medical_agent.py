from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from api.dependencies.answering import get_answering_service
from api.dependencies.appointments import get_appointment_graph
from guardrails.emergency import (
    DEFAULT_EMERGENCY_RULES,
    EmergencyGuardrail,
)
from integrations.nvidia_nim import get_nvidia_nim_chat_model
from services.appointment_workflow import AppointmentWorkflow
from services.intent_router import IntentRouter
from services.medical_agent import MedicalAgent


def get_medical_agent(
    appointment_graph: Annotated[
        CompiledStateGraph,
        Depends(get_appointment_graph),
    ],
) -> MedicalAgent:
    return MedicalAgent(
        emergency_guardrail=EmergencyGuardrail(
            rules=DEFAULT_EMERGENCY_RULES,
        ),
        intent_router=IntentRouter(
            chat_model=get_nvidia_nim_chat_model(),
        ),
        answering_service=get_answering_service(),
        appointment_workflow=AppointmentWorkflow(
            graph=appointment_graph,
        ),
    )
