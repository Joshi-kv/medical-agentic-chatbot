from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from services.answering import SAFETY_NOTICE, AnswerResult
from services.human_review import ReviewDecision


class AppointmentWorkflowError(Exception):
    """Custom exception for appointment workflow errors."""


class AppointmentWorkflow:
    def __init__(
        self,
        graph: CompiledStateGraph,
    ) -> None:
        self._graph = graph

    async def has_active_request(self, tenant_id: str, conversation_id: str) -> bool:

        snapshot = await self._graph.aget_state(
            self._config(tenant_id=tenant_id, conversation_id=conversation_id)
        )

        request_id = snapshot.values.get("appointment_request_id")
        appointment_id = snapshot.values.get("appointment_id")
        review_decision = snapshot.values.get("review_decision")

        return bool(
            request_id and appointment_id is None and review_decision != "rejected"
        )

    async def run(
        self,
        tenant_id: str,
        conversation_id: str,
        run_id: str,
        query: str,
    ) -> AnswerResult:
        config = self._config(tenant_id=tenant_id, conversation_id=conversation_id)

        snapshot = await self._graph.aget_state(config)

        paused_result = self._paused_result(snapshot.tasks)

        if paused_result is not None:
            return paused_result

        result = await self._graph.ainvoke(
            {
                "tenant_id": tenant_id,
                "conversation_id": conversation_id,
                "run_id": run_id,
                "query": query,
            },
            config=config,
        )

        interrupts = result.get("__interrupt__", ())
        if interrupts:
            return self._result_from_interrupt(interrupts[0])

        response = result.get("response")

        if not isinstance(response, str) or not response.strip():
            raise AppointmentWorkflowError(
                "Appointment workflow response is missing or empty."
            )

        review_case_id = result.get("review_case_id")

        if review_case_id is not None and not isinstance(review_case_id, str):
            raise AppointmentWorkflowError(
                "Appointment workflow review case id is not a string."
            )

        return self._answer(
            message=response,
            review_case_id=review_case_id,
        )

    async def resume(
        self,
        tenant_id: str,
        conversation_id: str,
        case_id: str,
        decision: ReviewDecision,
    ) -> AnswerResult:
        config = self._config(tenant_id=tenant_id, conversation_id=conversation_id)

        snapshot = await self._graph.aget_state(config)
        interrupt = self._first_interrupt(snapshot.tasks)

        if interrupt is not None:
            self._validate_interrupt_case(interrupt=interrupt, case_id=case_id)

            result = await self._graph.ainvoke(
                Command(resume={"decision": decision}), config=config
            )

        elif snapshot.next:
            result = await self._graph.ainvoke(None, config=config)
        else:
            result = snapshot.values

        return self._completed_result(result=result, case_id=case_id, decision=decision)

    def _paused_result(
        self,
        tasks: tuple[object, ...],
    ) -> AnswerResult | None:
        interrupt = self._first_interrupt(tasks)

        if interrupt is None:
            return None

        return self._result_from_interrupt(interrupt)

    def _result_from_interrupt(
        self,
        interrupt: Interrupt,
    ) -> AnswerResult:
        payload = interrupt.value

        if not isinstance(payload, dict):
            raise AppointmentWorkflowError(
                "Appointment interrupt has an invalid payload."
            )

        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise AppointmentWorkflowError("Appointment interrupt has no message.")

        case_id = payload.get("case_id")
        if not isinstance(case_id, str):
            case_id = None

        return self._answer(
            message=message,
            review_case_id=case_id,
        )

    def _answer(
        self,
        message: str,
        review_case_id: str | None,
    ) -> AnswerResult:
        return AnswerResult(
            answer=message,
            citations=[],
            sufficient_evidence=False,
            safety_notice=SAFETY_NOTICE,
            route="appointment_request",
            review_case_id=review_case_id,
        )

    def _config(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": (f"{tenant_id}:{conversation_id}")}}

    def _first_interrupt(
        self,
        tasks: tuple[object, ...],
    ) -> Interrupt | None:
        for task in tasks:
            interrupts = getattr(task, "interrupts", ())

            if interrupts:
                return interrupts[0]

        return None

    def _validate_interrupt_case(
        self,
        interrupt: Interrupt,
        case_id: str,
    ) -> None:
        payload = interrupt.value

        if not isinstance(payload, dict):
            raise AppointmentWorkflowError(
                "Appointment interrupt has an invalid payload."
            )

        interrupted_case_id = payload.get("case_id")

        if interrupted_case_id != case_id:
            raise AppointmentWorkflowError(
                "Review case does not match the paused appointment workflow."
            )

    def _completed_result(
        self,
        result: dict[str, object],
        case_id: str,
        decision: ReviewDecision,
    ) -> AnswerResult:
        result_case_id = result.get("review_case_id")

        if result_case_id != case_id:
            raise AppointmentWorkflowError(
                "Completed appointment workflow has an unexpected review case."
            )

        result_decision = result.get("review_decision")

        if result_decision != decision:
            raise AppointmentWorkflowError(
                "Completed appointment workflow has an unexpected review decision."
            )

        response = result.get("response")

        if not isinstance(response, str) or not response.strip():
            raise AppointmentWorkflowError(
                "Completed appointment workflow has no response."
            )

        return self._answer(
            message=response,
            review_case_id=case_id,
        )
