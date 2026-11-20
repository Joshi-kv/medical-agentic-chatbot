import asyncio
import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from prompts.appointments import (
    APPOINTMENT_EXTRACTION_SYSTEM_PROMPT,
    APPOINTMENT_EXTRACTION_USER_PROMPT,
)

EXTRACTION_TIMEOUT_SECONDS = 15


class ExtractedAppointmentDetails(BaseModel):
    department: str | None = Field(
        default=None,
        max_length=100,
    )
    preferred_date: str | None = Field(
        default=None,
        max_length=100,
    )
    preferred_time: str | None = Field(
        default=None,
        max_length=100,
    )
    contact: str | None = Field(
        default=None,
        max_length=200,
    )
    reason: str | None = Field(
        default=None,
        max_length=500,
    )
    consent_given: bool | None = None


class AppointmentExtractionError(Exception):
    """Raised when appointment details cannot be extracted."""


class AppointmentExtractor:
    def __init__(self, chat_model: BaseChatModel):
        self._chat_model = chat_model
        self._parser = PydanticOutputParser(pydantic_object=ExtractedAppointmentDetails)

    async def extract(
        self,
        patient_message: str,
        current_details: dict[str, object],
        previous_assistant_message: str | None = None,
    ) -> ExtractedAppointmentDetails:
        normalized_message = patient_message.strip()

        if not normalized_message:
            raise AppointmentExtractionError("Empty patient message provided")

        prompt = APPOINTMENT_EXTRACTION_USER_PROMPT.format(
            current_details=json.dumps(
                current_details,
                default=str,
                ensure_ascii=False,
            ),
            previous_assistant_message=(
                previous_assistant_message or "No previous assistant message."
            ),
            patient_message=normalized_message,
            format_instructions=(self._parser.get_format_instructions()),
        )

        try:
            async with asyncio.timeout(EXTRACTION_TIMEOUT_SECONDS):
                response = await self._chat_model.ainvoke(
                    [
                        SystemMessage(content=APPOINTMENT_EXTRACTION_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
        except TimeoutError as timeout_error:
            raise AppointmentExtractionError(
                f"Appointment extraction timeout after "
                f"{EXTRACTION_TIMEOUT_SECONDS} seconds"
            ) from timeout_error
        except Exception as exception:
            raise AppointmentExtractionError(
                "Failed to extract appointment details"
            ) from exception

        if not isinstance(response.content, str):
            raise AppointmentExtractionError(
                "Invalid response format from language model"
            )
        try:
            return self._parser.parse(response.content)
        except Exception as exception:
            raise AppointmentExtractionError(
                "Failed to parse appointment details from language model response"
            ) from exception
