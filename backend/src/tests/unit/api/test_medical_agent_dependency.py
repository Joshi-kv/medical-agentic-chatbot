from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.workflows.medical_agent.nodes import MedicalAgentNodes
from api.dependencies.medical_agent import build_medical_agent_nodes, get_medical_agent
from services.medical_agent import MedicalAgent


def test_build_medical_agent_nodes_wires_services() -> None:
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
        nodes = build_medical_agent_nodes(appointment_graph=appointment_graph)

        assert isinstance(nodes, MedicalAgentNodes)
        assert nodes._intent_router._chat_model is model
        assert nodes._answering_service is answering
        assert nodes._appointment_workflow._graph is appointment_graph


def test_get_medical_agent_uses_app_graph() -> None:
    graph = MagicMock()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        medical_agent_graph=graph
    )))

    agent = get_medical_agent(request=request)

    assert isinstance(agent, MedicalAgent)
    assert agent._graph is graph
