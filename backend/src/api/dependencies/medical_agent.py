from fastapi import HTTPException, Request, status
from langgraph.graph.state import CompiledStateGraph

from agent.workflows.medical_agent.nodes import MedicalAgentNodes
from api.dependencies.answering import get_answering_service
from guardrails.emergency import DEFAULT_EMERGENCY_RULES, EmergencyGuardrail
from integrations.nvidia_nim import get_nvidia_nim_chat_model
from services.appointment_workflow import AppointmentWorkflow
from services.conversation_context import ConversationContextualizer
from services.intent_router import IntentRouter
from services.medical_agent import MedicalAgent


def build_medical_agent_nodes(
    appointment_graph: CompiledStateGraph,
) -> MedicalAgentNodes:
    chat_model = get_nvidia_nim_chat_model()
    return MedicalAgentNodes(
        emergency_guardrail=EmergencyGuardrail(rules=DEFAULT_EMERGENCY_RULES),
        intent_router=IntentRouter(chat_model=chat_model),
        answering_service=get_answering_service(),
        appointment_workflow=AppointmentWorkflow(graph=appointment_graph),
        contextualizer=ConversationContextualizer(chat_model=chat_model),
    )


def get_medical_agent(request: Request) -> MedicalAgent:
    graph = getattr(request.app.state, "medical_agent_graph", None)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "medical_agent_unavailable",
                "message": "The medical agent is unavailable.",
            },
        )
    return MedicalAgent(graph=graph)
