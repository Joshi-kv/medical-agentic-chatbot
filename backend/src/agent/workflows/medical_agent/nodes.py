from uuid import UUID

from langgraph.config import get_stream_writer

from agent.workflows.medical_agent.state import (
    ContextMessageState,
    MedicalAgentState,
    answer_result_to_state,
)
from guardrails.emergency import EmergencyGuardrail
from guardrails.models import AgentRoute, SafetyAction
from services.answering import (
    SAFETY_NOTICE,
    AnswerCompleted,
    AnswerDelta,
    AnsweringService,
    AnswerProgress,
    AnswerResult,
)
from services.appointment_workflow import (
    AppointmentWorkflow,
    AppointmentWorkflowError,
)
from services.conversation_context import (
    ContextMessage,
    ConversationContextError,
    ConversationContextualizer,
)
from services.intent_router import IntentRouter, IntentRouterError

HOSPITAL_NAVIGATION_MESSAGE = (
    "I can help you find the appropriate hospital service. "
    "Which hospital location or department are you looking for?"
)

HUMAN_REVIEW_MESSAGE = (
    "This request needs review by qualified hospital staff. "
    "I can help collect the details needed for a handoff."
)

ROUTING_FAILURE_MESSAGE = (
    "I couldn't safely determine how to handle this request. "
    "Please provide more information or ask for hospital staff assistance."
)


class MedicalAgentNodes:
    def __init__(
        self,
        emergency_guardrail: EmergencyGuardrail,
        intent_router: IntentRouter,
        answering_service: AnsweringService,
        appointment_workflow: AppointmentWorkflow,
        contextualizer: ConversationContextualizer,
    ) -> None:
        self._emergency_guardrail = emergency_guardrail
        self._intent_router = intent_router
        self._answering_service = answering_service
        self._appointment_workflow = appointment_workflow
        self._contextualizer = contextualizer

    async def check_initial_safety(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        self._progress("safety", "Checking safety requirements...")

        decision = self._emergency_guardrail.evaluate(
            self._required_string(state, "query")
        )
        if decision.action != SafetyAction.EMERGENCY:
            return {}

        return self._complete(
            self._static_result(
                message=decision.response or "Seek emergency medical help now.",
                route="emergency",
            )
        )

    async def check_active_appointment(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        conversation_id = state.get("conversation_id")
        if not conversation_id:
            return {"active_appointment": False}

        active = await self._appointment_workflow.has_active_request(
            tenant_id=self._required_string(state, "tenant_id"),
            conversation_id=conversation_id,
        )
        return {"active_appointment": active}

    async def contextualize_query(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        query = self._required_string(state, "query")
        history_data = state.get("conversation_history", [])
        if not history_data:
            return {"standalone_query": query}

        self._progress("context", "Understanding the conversation context...")

        try:
            standalone_query = await self._contextualizer.rewrite(
                query=query,
                history=self._context_messages(history_data),
            )
        except ConversationContextError:
            standalone_query = query

        return {"standalone_query": standalone_query}

    async def check_contextualized_safety(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        query = state.get("standalone_query") or self._required_string(
            state, "query"
        )
        decision = self._emergency_guardrail.evaluate(query)
        if decision.action != SafetyAction.EMERGENCY:
            return {}

        return self._complete(
            self._static_result(
                message=decision.response or "Seek emergency medical help now.",
                route="emergency",
            )
        )

    async def classify_intent(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        self._progress("routing", "Selecting the appropriate hospital service...")

        query = state.get("standalone_query") or self._required_string(
            state, "query"
        )
        try:
            decision = await self._intent_router.classify(query)
        except IntentRouterError:
            return self._complete(
                self._static_result(
                    message=ROUTING_FAILURE_MESSAGE,
                    route=AgentRoute.HUMAN_REVIEW,
                )
            )

        return {
            "intent": decision.route.value,
            "follow_up_question": decision.follow_up_question,
        }

    async def answer_medical_question(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        query = state.get("standalone_query") or self._required_string(
            state, "query"
        )

        if not state.get("streaming"):
            result = await self._answering_service.answer(
                query=query,
                tenant_id=self._required_string(state, "tenant_id"),
                document_id=self._document_id(state),
            )
            return self._complete(result)

        final_result: AnswerResult | None = None

        async for event in self._answering_service.stream_answer(
            query=query,
            tenant_id=self._required_string(state, "tenant_id"),
            document_id=self._document_id(state),
        ):
            if isinstance(event, AnswerProgress):
                self._progress(event.stage, event.message)
            elif isinstance(event, AnswerDelta):
                self._write({"type": "answer_delta", "text": event.text})
            elif isinstance(event, AnswerCompleted):
                final_result = event.result

        if final_result is None:
            raise RuntimeError("Answering service completed without a result.")

        update = answer_result_to_state(final_result)
        self._write({"type": "answer_completed", "result": update})
        return update

    async def run_appointment(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        message = (
            "Continuing your appointment request..."
            if state.get("active_appointment")
            else "Starting your appointment request..."
        )
        self._progress("appointment", message)

        conversation_id = state.get("conversation_id")
        run_id = state.get("run_id")
        if not conversation_id or not run_id:
            raise AppointmentWorkflowError(
                "Conversation and run IDs are required for appointments."
            )

        result = await self._appointment_workflow.run(
            tenant_id=self._required_string(state, "tenant_id"),
            conversation_id=conversation_id,
            run_id=run_id,
            query=state.get("standalone_query")
            or self._required_string(state, "query"),
        )
        return self._complete(result)

    async def prepare_static_response(
        self,
        state: MedicalAgentState,
    ) -> dict[str, object]:
        intent = state.get("intent")
        follow_up = state.get("follow_up_question")

        if intent == AgentRoute.CLARIFICATION:
            message = follow_up or "Could you provide more information?"
        elif intent == AgentRoute.HOSPITAL_NAVIGATION:
            message = HOSPITAL_NAVIGATION_MESSAGE
        else:
            message = HUMAN_REVIEW_MESSAGE

        return self._complete(
            self._static_result(
                message=message,
                route=intent or AgentRoute.HUMAN_REVIEW,
            )
        )

    def _complete(self, result: AnswerResult) -> dict[str, object]:
        update = answer_result_to_state(result)
        self._write({"type": "answer_delta", "text": result.answer})
        self._write({"type": "answer_completed", "result": update})
        return update

    def _progress(self, stage: str, message: str) -> None:
        self._write({"type": "progress", "stage": stage, "message": message})

    def _write(self, event: dict[str, object]) -> None:
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return
        writer(event)

    def _static_result(self, message: str, route: str) -> AnswerResult:
        return AnswerResult(
            answer=message,
            citations=[],
            sufficient_evidence=False,
            safety_notice=SAFETY_NOTICE,
            route=str(route),
        )

    def _context_messages(
        self,
        history: list[ContextMessageState],
    ) -> list[ContextMessage]:
        return [
            ContextMessage(role=message["role"], content=message["content"])
            for message in history
        ]

    def _document_id(self, state: MedicalAgentState) -> UUID | None:
        value = state.get("document_id")
        return UUID(value) if value else None

    def _required_string(self, state: MedicalAgentState, field: str) -> str:
        value = state.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Medical-agent state is missing {field}.")
        return value.strip()
