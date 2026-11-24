from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.workflows.medical_agent.nodes import MedicalAgentNodes
from agent.workflows.medical_agent.state import MedicalAgentState
from guardrails.models import AgentRoute


def route_after_safety(state: MedicalAgentState) -> str:
    return "done" if state.get("answer") else "continue"


def route_after_appointment_check(state: MedicalAgentState) -> str:
    return "appointment" if state.get("active_appointment") else "context"


def route_after_intent(state: MedicalAgentState) -> str:
    if state.get("answer"):
        return "done"

    intent = state.get("intent")
    if intent == AgentRoute.EDUCATION:
        return "education"
    if intent == AgentRoute.APPOINTMENT_REQUEST:
        return "appointment"
    return "static_response"


def build_medical_agent_graph(
    nodes: MedicalAgentNodes,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    builder = StateGraph(MedicalAgentState)

    builder.add_node("check_initial_safety", nodes.check_initial_safety)
    builder.add_node("check_active_appointment", nodes.check_active_appointment)
    builder.add_node("contextualize_query", nodes.contextualize_query)
    builder.add_node(
        "check_contextualized_safety", nodes.check_contextualized_safety
    )
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("answer_medical_question", nodes.answer_medical_question)
    builder.add_node("run_appointment", nodes.run_appointment)
    builder.add_node("prepare_static_response", nodes.prepare_static_response)

    builder.add_edge(START, "check_initial_safety")
    builder.add_conditional_edges(
        "check_initial_safety",
        route_after_safety,
        {"continue": "check_active_appointment", "done": END},
    )
    builder.add_conditional_edges(
        "check_active_appointment",
        route_after_appointment_check,
        {"appointment": "run_appointment", "context": "contextualize_query"},
    )
    builder.add_edge("contextualize_query", "check_contextualized_safety")
    builder.add_conditional_edges(
        "check_contextualized_safety",
        route_after_safety,
        {"continue": "classify_intent", "done": END},
    )
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "education": "answer_medical_question",
            "appointment": "run_appointment",
            "static_response": "prepare_static_response",
            "done": END,
        },
    )
    builder.add_edge("answer_medical_question", END)
    builder.add_edge("run_appointment", END)
    builder.add_edge("prepare_static_response", END)

    return builder.compile(checkpointer=checkpointer, name="medical_agent")
