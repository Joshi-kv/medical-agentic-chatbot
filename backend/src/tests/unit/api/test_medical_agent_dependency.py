from unittest.mock import MagicMock, patch

from api.dependencies.medical_agent import get_medical_agent
from services.medical_agent import MedicalAgent


def test_get_medical_agent_wires_services() -> None:
    model = MagicMock()
    answering = MagicMock()
    appointment_graph = MagicMock()

    with (
        patch(
            "api.dependencies.medical_agent.get_nvidia_nim_chat_model",
            return_value=model,
        ),
        patch(
            "api.dependencies.medical_agent.get_answering_service",
            return_value=answering,
        ),
    ):
        agent = get_medical_agent(appointment_graph=appointment_graph)

        assert isinstance(agent, MedicalAgent)
        assert agent._intent_router._chat_model is model
        assert agent._answering_service is answering
        assert agent._appointment_workflow._graph is appointment_graph
