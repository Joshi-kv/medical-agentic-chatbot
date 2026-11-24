from agent.workflows.medical_agent.graph import (
    route_after_appointment_check,
    route_after_intent,
    route_after_safety,
)


def test_graph_routes_emergency_result_directly_to_end() -> None:
    assert route_after_safety({"answer": "Seek emergency help now."}) == "done"
    assert route_after_safety({}) == "continue"


def test_graph_routes_active_appointment_before_general_context() -> None:
    assert route_after_appointment_check({"active_appointment": True}) == (
        "appointment"
    )
    assert route_after_appointment_check({"active_appointment": False}) == "context"


def test_graph_routes_supported_intents() -> None:
    assert route_after_intent({"intent": "education"}) == "education"
    assert route_after_intent({"intent": "appointment_request"}) == "appointment"
    assert route_after_intent({"intent": "clarification"}) == "static_response"
