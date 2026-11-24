from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.workflows.appointments.nodes import AppointmentNodes
from agent.workflows.appointments.state import MedicalAgentState


def build_appointment_graph(
    nodes: AppointmentNodes,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build the multi-turn appointment workflow"""

    builder = StateGraph(MedicalAgentState)

    builder.add_node("ensure_appointment_request", nodes.ensure_appointment_request)
    builder.add_node("collect_appointment_details", nodes.collect_appointment_details)
    builder.add_node(
        "ask_next_appointment_question", nodes.ask_next_appointment_question
    )
    builder.add_node("request_appointment_review", nodes.request_appointment_review)
    builder.add_node("book_appointment", nodes.book_appointment)
    builder.add_node("appointment_rejected", nodes.appointment_rejected)

    builder.add_edge(START, "ensure_appointment_request")
    builder.add_edge("ensure_appointment_request", "collect_appointment_details")
    builder.add_conditional_edges(
        "collect_appointment_details",
        nodes.choose_next_appointment_step,
        {
            "ask_next_appointment_question": ("ask_next_appointment_question"),
            "request_appointment_review": ("request_appointment_review"),
        },
    )
    builder.add_edge("ask_next_appointment_question", END)
    builder.add_conditional_edges(
        "request_appointment_review",
        nodes.choose_after_appointment_review,
        {
            "book_appointment": "book_appointment",
            "appointment_rejected": ("appointment_rejected"),
        },
    )

    builder.add_edge(
        "book_appointment",
        END,
    )
    builder.add_edge(
        "appointment_rejected",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer,
        name="appointment_workflow",
    )
