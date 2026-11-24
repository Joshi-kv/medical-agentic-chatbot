from collections.abc import AsyncIterator
from uuid import UUID

from guardrails.emergency import EmergencyGuardrail
from guardrails.models import AgentRoute, SafetyAction
from services.answering import (
    SAFETY_NOTICE,
    AnswerCompleted,
    AnswerDelta,
    AnsweringService,
    AnswerProgress,
    AnswerResult,
    AnswerStreamItem,
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


class MedicalAgent:
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

    async def answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        conversation_history: list[ContextMessage] | None = None,
    ) -> AnswerResult:
        safety = self._emergency_guardrail.evaluate(query)

        if safety.action == SafetyAction.EMERGENCY:
            return self._static_result(
                message=(safety.response or "Seek emergency medical help now."),
                route="emergency",
            )

        if await self._has_active_appointment(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        ):
            return await self._run_appointment(
                query=query,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )

        standalone_query = await self._standalone_query(
            query=query,
            history=conversation_history,
        )
        contextualized_safety = self._emergency_guardrail.evaluate(standalone_query)

        if contextualized_safety.action == SafetyAction.EMERGENCY:
            return self._static_result(
                message=(
                    contextualized_safety.response or "Seek emergency medical help now."
                ),
                route="emergency",
            )

        try:
            decision = await self._intent_router.classify(standalone_query)
        except IntentRouterError:
            return self._static_result(
                message=ROUTING_FAILURE_MESSAGE,
                route=AgentRoute.HUMAN_REVIEW,
            )

        if decision.route == AgentRoute.EDUCATION:
            return await self._answering_service.answer(
                query=standalone_query,
                tenant_id=tenant_id,
                document_id=document_id,
            )

        if decision.route == AgentRoute.APPOINTMENT_REQUEST:
            return await self._run_appointment(
                query=standalone_query,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )

        return self._result_for_route(
            route=decision.route,
            follow_up_question=decision.follow_up_question,
        )

    async def stream_answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        conversation_history: list[ContextMessage] | None = None,
    ) -> AsyncIterator[AnswerStreamItem]:
        yield AnswerProgress(
            stage="safety",
            message="Checking safety requirements...",
        )

        safety = self._emergency_guardrail.evaluate(query)

        if safety.action == SafetyAction.EMERGENCY:
            result = self._static_result(
                message=(safety.response or "Seek emergency medical help now."),
                route="emergency",
            )
            yield AnswerDelta(text=result.answer)
            yield AnswerCompleted(result=result)
            return

        active_appointment = await self._has_active_appointment(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

        if active_appointment:
            yield AnswerProgress(
                stage="appointment",
                message="Continuing your appointment request...",
            )

            result = await self._run_appointment(
                query=query,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=run_id,
            )

            yield AnswerDelta(text=result.answer)
            yield AnswerCompleted(result=result)
            return

        if conversation_history:
            yield AnswerProgress(
                stage="context",
                message="Understanding the conversation context...",
            )

        standalone_query = await self._standalone_query(
            query=query,
            history=conversation_history,
        )
        contextualized_safety = self._emergency_guardrail.evaluate(standalone_query)

        if contextualized_safety.action == SafetyAction.EMERGENCY:
            result = self._static_result(
                message=(
                    contextualized_safety.response or "Seek emergency medical help now."
                ),
                route="emergency",
            )
            yield AnswerDelta(text=result.answer)
            yield AnswerCompleted(result=result)
            return

        yield AnswerProgress(
            stage="routing",
            message="Selecting the appropriate hospital service...",
        )

        try:
            decision = await self._intent_router.classify(standalone_query)
        except IntentRouterError:
            result = self._static_result(
                message=ROUTING_FAILURE_MESSAGE,
                route=AgentRoute.HUMAN_REVIEW,
            )
        else:
            if decision.route == AgentRoute.EDUCATION:
                async for event in self._answering_service.stream_answer(
                    query=standalone_query,
                    tenant_id=tenant_id,
                    document_id=document_id,
                ):
                    yield event
                return

            if decision.route == AgentRoute.APPOINTMENT_REQUEST:
                yield AnswerProgress(
                    stage="appointment",
                    message="Starting your appointment request...",
                )

                result = await self._run_appointment(
                    query=standalone_query,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    run_id=run_id,
                )
            else:
                result = self._result_for_route(
                    route=decision.route,
                    follow_up_question=decision.follow_up_question,
                )

        yield AnswerDelta(text=result.answer)
        yield AnswerCompleted(result=result)

    async def _has_active_appointment(
        self,
        tenant_id: str,
        conversation_id: str | None,
    ) -> bool:
        if conversation_id is None:
            return False

        return await self._appointment_workflow.has_active_request(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
        )

    async def _standalone_query(
        self,
        query: str,
        history: list[ContextMessage] | None,
    ) -> str:
        if not history:
            return query

        try:
            return await self._contextualizer.rewrite(
                query=query,
                history=history,
            )
        except ConversationContextError:
            return query

    async def _run_appointment(
        self,
        query: str,
        tenant_id: str,
        conversation_id: str | None,
        run_id: str | None,
    ) -> AnswerResult:
        if conversation_id is None or run_id is None:
            raise AppointmentWorkflowError(
                "Conversation and run IDs are required for appointment requests."
            )

        return await self._appointment_workflow.run(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            run_id=run_id,
            query=query,
        )

    def _result_for_route(
        self,
        route: AgentRoute,
        follow_up_question: str | None,
    ) -> AnswerResult:
        if route == AgentRoute.CLARIFICATION:
            message = follow_up_question or "Could you provide more information?"
        elif route == AgentRoute.HOSPITAL_NAVIGATION:
            message = HOSPITAL_NAVIGATION_MESSAGE
        else:
            message = HUMAN_REVIEW_MESSAGE

        return self._static_result(
            message=message,
            route=route,
        )

    def _static_result(
        self,
        message: str,
        route: str,
    ) -> AnswerResult:
        return AnswerResult(
            answer=message,
            citations=[],
            sufficient_evidence=False,
            safety_notice=SAFETY_NOTICE,
            route=str(route),
        )
