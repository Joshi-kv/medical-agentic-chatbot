from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from services.answering import SAFETY_NOTICE, AnswerResult
from services.appointment_workflow import (
    AppointmentWorkflow,
    AppointmentWorkflowError,
)


def make_snapshot(
    values: dict[str, object] | None = None,
    tasks: tuple[object, ...] = (),
    next_nodes: tuple[str, ...] = (),
) -> MagicMock:
    snapshot = MagicMock()
    snapshot.values = values if values is not None else {}
    snapshot.tasks = tasks
    snapshot.next = next_nodes
    return snapshot


def make_task_with_interrupt(value: object) -> MagicMock:
    task = MagicMock()
    task.interrupts = [Interrupt(value=value)]
    return task


@pytest.mark.anyio
async def test_has_active_request_returns_true_when_in_progress() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        values={
            "appointment_request_id": "req-123",
            "appointment_id": None,
            "review_decision": None,
        }
    )
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.has_active_request(
        tenant_id="tenant-1",
        conversation_id="conv-1",
    )

    assert result is True
    graph.aget_state.assert_awaited_once_with(
        {"configurable": {"thread_id": "tenant-1:conv-1"}}
    )


@pytest.mark.anyio
async def test_has_active_request_returns_false_when_booked() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        values={
            "appointment_request_id": "req-123",
            "appointment_id": "apt-999",
            "review_decision": "approved",
        }
    )
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.has_active_request(
        tenant_id="tenant-1",
        conversation_id="conv-1",
    )

    assert result is False


@pytest.mark.anyio
async def test_has_active_request_returns_false_when_rejected() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        values={
            "appointment_request_id": "req-123",
            "appointment_id": None,
            "review_decision": "rejected",
        }
    )
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.has_active_request(
        tenant_id="tenant-1",
        conversation_id="conv-1",
    )

    assert result is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "values",
    [
        {},
        {"appointment_request_id": None, "appointment_id": None},
        {"appointment_request_id": ""},
    ],
)
async def test_has_active_request_returns_false_when_no_request(
    values: dict[str, object],
) -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(values=values)
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.has_active_request(
        tenant_id="tenant-1",
        conversation_id="conv-1",
    )

    assert result is False


@pytest.mark.anyio
async def test_run_returns_paused_result_when_graph_already_interrupted() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    task = make_task_with_interrupt(
        value={
            "message": "Your request is awaiting hospital staff review.",
            "case_id": "case-456",
        }
    )
    graph.aget_state.return_value = make_snapshot(tasks=(task,))
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.run(
        tenant_id="tenant-1",
        conversation_id="conv-1",
        run_id="run-1",
        query="Any updates?",
    )

    assert isinstance(result, AnswerResult)
    assert result.answer == "Your request is awaiting hospital staff review."
    assert result.review_case_id == "case-456"
    assert result.route == "appointment_request"
    assert result.sufficient_evidence is False
    assert result.citations == []
    assert result.safety_notice == SAFETY_NOTICE

    graph.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_run_paused_result_handles_missing_or_non_string_case_id() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    task = make_task_with_interrupt(
        value={"message": "Awaiting review.", "case_id": None}
    )
    graph.aget_state.return_value = make_snapshot(tasks=(task,))
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.run(
        tenant_id="tenant-1",
        conversation_id="conv-1",
        run_id="run-1",
        query="Status?",
    )

    assert result.review_case_id is None
    assert result.answer == "Awaiting review."


@pytest.mark.anyio
async def test_run_success_when_asking_question_without_review_case() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {
        "response": "Which hospital department or type of care do you need?",
        "review_case_id": None,
    }
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.run(
        tenant_id="tenant-1",
        conversation_id="conv-1",
        run_id="run-1",
        query="I want to see a doctor",
    )

    graph.ainvoke.assert_awaited_once_with(
        {
            "tenant_id": "tenant-1",
            "conversation_id": "conv-1",
            "run_id": "run-1",
            "query": "I want to see a doctor",
        },
        config={"configurable": {"thread_id": "tenant-1:conv-1"}},
    )
    assert result.answer == "Which hospital department or type of care do you need?"
    assert result.review_case_id is None
    assert result.route == "appointment_request"


@pytest.mark.anyio
async def test_run_success_with_review_case_id() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {
        "response": "Your appointment request has been scheduled.",
        "review_case_id": "case-999",
    }
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.run(
        tenant_id="tenant-1",
        conversation_id="conv-1",
        run_id="run-1",
        query="Confirmed",
    )

    assert result.answer == "Your appointment request has been scheduled."
    assert result.review_case_id == "case-999"


@pytest.mark.anyio
async def test_run_returns_result_from_new_interrupt_during_ainvoke() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {
        "__interrupt__": [
            Interrupt(
                value={
                    "message": "Sent for staff review.",
                    "case_id": "case-new",
                }
            )
        ]
    }
    workflow = AppointmentWorkflow(graph=graph)

    result = await workflow.run(
        tenant_id="tenant-1",
        conversation_id="conv-1",
        run_id="run-1",
        query="Yes I consent",
    )

    assert result.answer == "Sent for staff review."
    assert result.review_case_id == "case-new"
    assert result.route == "appointment_request"


@pytest.mark.anyio
@pytest.mark.parametrize("invalid_response", [None, "", "   ", 12345, [], {}])
async def test_run_raises_when_response_is_not_string(
    invalid_response: object,
) -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {"response": invalid_response}
    workflow = AppointmentWorkflow(graph=graph)

    with pytest.raises(
        AppointmentWorkflowError,
        match="Appointment workflow response is missing or empty.",
    ):
        await workflow.run(
            tenant_id="tenant-1",
            conversation_id="conv-1",
            run_id="run-1",
            query="Hello",
        )


@pytest.mark.anyio
@pytest.mark.parametrize("invalid_case_id", [123, [], {}, True])
async def test_run_raises_when_review_case_id_is_invalid_type(
    invalid_case_id: object,
) -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {
        "response": "Valid message",
        "review_case_id": invalid_case_id,
    }
    workflow = AppointmentWorkflow(graph=graph)

    with pytest.raises(
        AppointmentWorkflowError,
        match="Appointment workflow review case id is not a string.",
    ):
        await workflow.run(
            tenant_id="tenant-1",
            conversation_id="conv-1",
            run_id="run-1",
            query="Hello",
        )


@pytest.mark.anyio
async def test_run_raises_when_interrupt_payload_is_not_dict() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {
        "__interrupt__": [Interrupt(value="not a dictionary")]
    }
    workflow = AppointmentWorkflow(graph=graph)

    with pytest.raises(
        AppointmentWorkflowError,
        match="Appointment interrupt has an invalid payload.",
    ):
        await workflow.run(
            tenant_id="tenant-1",
            conversation_id="conv-1",
            run_id="run-1",
            query="Hello",
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"case_id": "case-1"},
        {"message": None},
        {"message": "   "},
    ],
)
async def test_run_raises_when_interrupt_message_is_missing_or_empty(
    payload: dict[str, object],
) -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(tasks=())
    graph.ainvoke.return_value = {"__interrupt__": [Interrupt(value=payload)]}
    workflow = AppointmentWorkflow(graph=graph)

    with pytest.raises(
        AppointmentWorkflowError,
        match="Appointment interrupt has no message.",
    ):
        await workflow.run(
            tenant_id="tenant-1",
            conversation_id="conv-1",
            run_id="run-1",
            query="Hello",
        )


@pytest.mark.anyio
async def test_resume_approved_review() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        tasks=(
            make_task_with_interrupt(
                {
                    "case_id": "case-1",
                    "message": "Waiting for staff review.",
                }
            ),
        )
    )
    graph.ainvoke.return_value = {
        "review_case_id": "case-1",
        "review_decision": "approved",
        "appointment_id": "appointment-1",
        "response": "Your appointment has been scheduled.",
    }
    workflow = AppointmentWorkflow(graph)

    result = await workflow.resume(
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        case_id="case-1",
        decision="approved",
    )

    assert result.answer == "Your appointment has been scheduled."
    command = graph.ainvoke.await_args.args[0]
    assert isinstance(command, Command)
    assert command.resume == {"decision": "approved"}
    assert graph.ainvoke.await_args.kwargs["config"] == {
        "configurable": {"thread_id": "tenant-1:conversation-1"}
    }


@pytest.mark.anyio
async def test_resume_rejects_wrong_review_case() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        tasks=(
            make_task_with_interrupt(
                {
                    "case_id": "different-case",
                    "message": "Waiting for staff review.",
                }
            ),
        )
    )
    workflow = AppointmentWorkflow(graph)

    with pytest.raises(AppointmentWorkflowError, match="does not match"):
        await workflow.resume(
            tenant_id="tenant-1",
            conversation_id="conversation-1",
            case_id="case-1",
            decision="approved",
        )

    graph.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_resume_is_idempotent_after_completion() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        values={
            "review_case_id": "case-1",
            "review_decision": "approved",
            "appointment_id": "appointment-1",
            "response": "Your appointment has been scheduled.",
        }
    )
    workflow = AppointmentWorkflow(graph)

    result = await workflow.resume(
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        case_id="case-1",
        decision="approved",
    )

    assert result.answer == "Your appointment has been scheduled."
    graph.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_resume_retries_unfinished_node_without_interrupt() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    graph.aget_state.return_value = make_snapshot(
        values={
            "review_case_id": "case-1",
            "review_decision": "approved",
        },
        next_nodes=("book_appointment",),
    )
    graph.ainvoke.return_value = {
        "review_case_id": "case-1",
        "review_decision": "approved",
        "appointment_id": "appointment-1",
        "response": "Your appointment has been scheduled.",
    }
    workflow = AppointmentWorkflow(graph)

    await workflow.resume(
        tenant_id="tenant-1",
        conversation_id="conversation-1",
        case_id="case-1",
        decision="approved",
    )

    graph.ainvoke.assert_awaited_once_with(
        None,
        config={"configurable": {"thread_id": "tenant-1:conversation-1"}},
    )


def test_config_helper() -> None:
    graph = AsyncMock(spec=CompiledStateGraph)
    workflow = AppointmentWorkflow(graph=graph)

    config = workflow._config(
        tenant_id="tenant-abc",
        conversation_id="conv-xyz",
    )

    assert config == {
        "configurable": {
            "thread_id": "tenant-abc:conv-xyz",
        }
    }
