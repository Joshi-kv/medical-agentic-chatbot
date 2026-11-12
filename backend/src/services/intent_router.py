import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from guardrails.models import AgentRoute, RouteDecision
from prompts.routing import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT

ROUTER_TIMEOUT_SECONDS = 15


class RouterOutput(BaseModel):
    route: AgentRoute
    confidence: float = Field(ge=0, le=1)
    follow_up_question: str | None = Field(default=None, max_length=300)


class IntentRouterError(Exception):
    pass


class IntentRouter:
    def __init__(
        self,
        chat_model: BaseChatModel,
        minimum_confidence: float = 0.65,
    ) -> None:
        if not 0 <= minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between 0 and 1.")

        self._chat_model = chat_model
        self._minimum_confidence = minimum_confidence
        self._parser = PydanticOutputParser(pydantic_object=RouterOutput)

    async def classify(self, message: str) -> RouteDecision:
        normalized_message = message.strip()
        if not normalized_message:
            return self._clarification("What would you like help with?", confidence=1)

        if len(normalized_message.split()) == 1:
            return self._clarification(
                "Could you tell me what you need help with?",
                confidence=1,
            )

        appointment_phrases = (
            "schedule an appointment",
            "book an appointment",
            "reschedule my appointment",
            "cancel my appointment",
        )
        if any(
            phrase in normalized_message.casefold() for phrase in appointment_phrases
        ):
            return RouteDecision(
                route=AgentRoute.APPOINTMENT_REQUEST,
                confidence=1,
            )

        prompt = ROUTER_USER_PROMPT.format(
            message=normalized_message,
            format_instructions=self._parser.get_format_instructions(),
        )

        try:
            async with asyncio.timeout(ROUTER_TIMEOUT_SECONDS):
                response = await self._chat_model.ainvoke(
                    [
                        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
            if not isinstance(response.content, str):
                raise IntentRouterError(
                    "Intent classifier returned unsupported content."
                )
            output = self._parser.parse(response.content)
        except IntentRouterError:
            raise
        except Exception as error:
            raise IntentRouterError("Intent classification failed.") from error

        if output.confidence < self._minimum_confidence:
            return self._clarification(
                "Do you need medical information, hospital directions, "
                "or help with an appointment?",
                confidence=output.confidence,
            )

        if output.route == AgentRoute.CLARIFICATION:
            return self._clarification(
                output.follow_up_question
                or "Could you provide a little more information about what you need?",
                confidence=output.confidence,
            )

        return RouteDecision(
            route=output.route,
            confidence=output.confidence,
        )

    def _clarification(self, question: str, confidence: float) -> RouteDecision:
        return RouteDecision(
            route=AgentRoute.CLARIFICATION,
            confidence=confidence,
            follow_up_question=question,
        )
