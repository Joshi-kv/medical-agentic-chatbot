"""Compiled graph exports used by the local LangGraph Studio server."""

from agent.workflows.appointments.graph import build_appointment_graph
from api.dependencies.appointments import get_appointment_nodes
from api.dependencies.web_search import get_web_search_graph

appointment_graph = build_appointment_graph(
    nodes=get_appointment_nodes(),
)

web_search_graph = get_web_search_graph()
