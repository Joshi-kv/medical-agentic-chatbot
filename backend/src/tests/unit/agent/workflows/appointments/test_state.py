from typing import get_args

from agent.workflows.appointments.state import AgentRoute, AppointmentState


def test_agent_route_literals() -> None:
    expected_routes = {
        "emergency",
        "education",
        "hospital_navigation",
        "appointment_request",
        "clarification",
        "human_review",
    }
    assert set(get_args(AgentRoute)) == expected_routes


def test_appointment_state_keys() -> None:
    expected_keys = {
        "tenant_id",
        "conversation_id",
        "run_id",
        "query",
        "previous_assistant_message",
        "route",
        "appointment_request_id",
        "missing_appointment_fields",
        "review_case_id",
        "review_decision",
        "appointment_id",
        "response",
    }

    assert AppointmentState.__optional_keys__ == expected_keys
    assert AppointmentState.__required_keys__ == frozenset()


def test_appointment_state_empty_instantiation() -> None:
    state: AppointmentState = {}
    assert state == {}


def test_appointment_state_partial_instantiation() -> None:
    state: AppointmentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-123",
        "run_id": "run-456",
        "query": "I need to see a cardiologist tomorrow.",
        "previous_assistant_message": None,
    }

    assert state["tenant_id"] == "tenant-1"
    assert state["conversation_id"] == "conv-123"
    assert state["run_id"] == "run-456"
    assert state["query"] == "I need to see a cardiologist tomorrow."
    assert state["previous_assistant_message"] is None
    assert "route" not in state


def test_appointment_state_full_instantiation() -> None:
    state: AppointmentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-123",
        "run_id": "run-456",
        "query": "I agree to submit.",
        "previous_assistant_message": (
            "Do you consent to submit this appointment request?"
        ),
        "route": "appointment_request",
        "appointment_request_id": "req-789",
        "missing_appointment_fields": [],
        "review_case_id": "case-999",
        "review_decision": "approved",
        "appointment_id": "app-111",
        "response": "Your appointment has been successfully scheduled.",
    }

    assert state["route"] == "appointment_request"
    assert state["appointment_request_id"] == "req-789"
    assert state["missing_appointment_fields"] == []
    assert state["review_case_id"] == "case-999"
    assert state["review_decision"] == "approved"
    assert state["appointment_id"] == "app-111"
    assert state["response"] == "Your appointment has been successfully scheduled."


def test_appointment_state_functional_update() -> None:
    initial_state: AppointmentState = {
        "tenant_id": "tenant-1",
        "conversation_id": "conv-123",
        "run_id": "run-456",
        "query": "What is hypertension?",
        "previous_assistant_message": None,
    }

    updated_state: AppointmentState = {
        **initial_state,
        "route": "education",
        "response": "Hypertension is high blood pressure [C1].",
    }

    assert updated_state["route"] == "education"
    assert updated_state["response"] == "Hypertension is high blood pressure [C1]."
    assert updated_state["query"] == "What is hypertension?"
    assert "route" not in initial_state
