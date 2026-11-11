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
from services.intent_router import IntentRouter, IntentRouterError

HOSPITAL_NAVIGATION_MESSAGE = (
    "I can help you find the appropriate hospital service. "
    "Which hospital location or department are you looking for?"
)
APPOINTMENT_MESSAGE = (
    "I can help collect information for an appointment request. "
    "Which department or type of care do you need?"
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
    ) -> None:
        self._emergency_guardrail = emergency_guardrail
        self._intent_router = intent_router
        self._answering_service = answering_service

    async def answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
    ) -> AnswerResult:
        safety = self._emergency_guardrail.evaluate(query)
        if safety.action == SafetyAction.EMERGENCY:
            return self._static_result(
                message=safety.response or "Seek emergency medical help now.",
                route="emergency",
            )

        try:
            decision = await self._intent_router.classify(query)
        except IntentRouterError:
            return self._static_result(
                message=ROUTING_FAILURE_MESSAGE,
                route=AgentRoute.HUMAN_REVIEW,
            )

        if decision.route == AgentRoute.EDUCATION:
            return await self._answering_service.answer(
                query=query,
                tenant_id=tenant_id,
                document_id=document_id,
            )

        return self._result_for_route(decision.route, decision.follow_up_question)

    async def stream_answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
    ) -> AsyncIterator[AnswerStreamItem]:
        yield AnswerProgress(stage="safety", message="Checking safety requirements...")

        safety = self._emergency_guardrail.evaluate(query)
        if safety.action == SafetyAction.EMERGENCY:
            result = self._static_result(
                message=safety.response or "Seek emergency medical help now.",
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
            decision = await self._intent_router.classify(query)
        except IntentRouterError:
            result = self._static_result(
                message=ROUTING_FAILURE_MESSAGE,
                route=AgentRoute.HUMAN_REVIEW,
            )
        else:
            if decision.route == AgentRoute.EDUCATION:
                async for event in self._answering_service.stream_answer(
                    query=query,
                    tenant_id=tenant_id,
                    document_id=document_id,
                ):
                    yield event
                return
            result = self._result_for_route(
                decision.route,
                decision.follow_up_question,
            )

        yield AnswerDelta(text=result.answer)
        yield AnswerCompleted(result=result)

    def _result_for_route(
        self,
        route: AgentRoute,
        follow_up_question: str | None,
    ) -> AnswerResult:
        if route == AgentRoute.CLARIFICATION:
            message = follow_up_question or "Could you provide more information?"
        elif route == AgentRoute.HOSPITAL_NAVIGATION:
            message = HOSPITAL_NAVIGATION_MESSAGE
        elif route == AgentRoute.APPOINTMENT_REQUEST:
            message = APPOINTMENT_MESSAGE
        else:
            message = HUMAN_REVIEW_MESSAGE

        return self._static_result(message=message, route=route)

    def _static_result(self, message: str, route: str) -> AnswerResult:
        return AnswerResult(
            answer=message,
            citations=[],
            sufficient_evidence=False,
            safety_notice=SAFETY_NOTICE,
            route=str(route),
        )
