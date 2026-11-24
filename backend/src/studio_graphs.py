"""Compiled graph exports used by the local LangGraph Studio server."""

from agent.workflows.appointments.graph import build_appointment_graph
from agent.workflows.medical_agent.graph import build_medical_agent_graph
from api.dependencies.appointments import get_appointment_nodes
from api.dependencies.medical_agent import build_medical_agent_nodes
from api.dependencies.web_search import get_web_search_graph

appointment_graph = build_appointment_graph(
    nodes=get_appointment_nodes(),
)

medical_agent_graph = build_medical_agent_graph(
    nodes=build_medical_agent_nodes(appointment_graph),
)

web_search_graph = get_web_search_graph()
