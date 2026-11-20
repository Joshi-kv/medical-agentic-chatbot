from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from services.appointment_extractor import (
    AppointmentExtractionError,
    AppointmentExtractor,
    ExtractedAppointmentDetails,
)


@pytest.mark.anyio
async def test_extract_all_fields_success() -> None:
    model = AsyncMock()
    json_content = (
        '{"department": "Cardiology", '
        '"preferred_date": "next Tuesday", '
        '"preferred_time": "morning", '
        '"contact": "+1-555-0100", '
        '"reason": "Chest tightness", '
        '"consent_given": true}'
    )
    model.ainvoke.return_value = AIMessage(content=json_content)
    extractor = AppointmentExtractor(chat_model=model)

    result = await extractor.extract(
        patient_message=(
            "I want to see Cardiology next Tuesday morning, "
            "call 555-0100. I agree."
        ),
        current_details={"department": None},
        previous_assistant_message="Which department would you like to visit?",
    )

    assert isinstance(result, ExtractedAppointmentDetails)
    assert result.department == "Cardiology"
    assert result.preferred_date == "next Tuesday"
    assert result.preferred_time == "morning"
    assert result.contact == "+1-555-0100"
    assert result.reason == "Chest tightness"
    assert result.consent_given is True

    model.ainvoke.assert_awaited_once()
    messages = model.ainvoke.await_args.args[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "Which department would you like to visit?" in messages[1].content
    assert "I want to see Cardiology" in messages[1].content


@pytest.mark.anyio
async def test_extract_partial_fields_defaults_none() -> None:
    model = AsyncMock()
    json_content = (
        '{"department": "Neurology", '
        '"preferred_date": null, '
        '"preferred_time": null, '
        '"contact": null, '
        '"reason": null, '
        '"consent_given": null}'
    )
    model.ainvoke.return_value = AIMessage(content=json_content)
    extractor = AppointmentExtractor(chat_model=model)

    result = await extractor.extract(
        patient_message="I need to see Neurology.",
        current_details={},
    )

    assert result.department == "Neurology"
    assert result.preferred_date is None
    assert result.preferred_time is None
    assert result.contact is None
    assert result.reason is None
    assert result.consent_given is None


@pytest.mark.anyio
async def test_extract_formats_current_details_and_previous_assistant_message() -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content='{"department": null, "preferred_date": "tomorrow", '
        '"preferred_time": null, "contact": null, "reason": null, '
        '"consent_given": null}'
    )
    extractor = AppointmentExtractor(chat_model=model)

    await extractor.extract(
        patient_message="Tomorrow works.",
        current_details={"department": "Dermatology", "consent_given": True},
        previous_assistant_message="When would you like your appointment?",
    )

    messages = model.ainvoke.await_args.args[0]
    user_prompt = messages[1].content
    assert '"department": "Dermatology"' in user_prompt
    assert '"consent_given": true' in user_prompt
    assert "When would you like your appointment?" in user_prompt


@pytest.mark.anyio
async def test_extract_uses_fallback_for_missing_previous_assistant_message() -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(
        content='{"department": null, "preferred_date": null, '
        '"preferred_time": null, "contact": null, "reason": null, '
        '"consent_given": null}'
    )
    extractor = AppointmentExtractor(chat_model=model)

    await extractor.extract(
        patient_message="Hello",
        current_details={},
        previous_assistant_message=None,
    )

    messages = model.ainvoke.await_args.args[0]
    user_prompt = messages[1].content
    assert "No previous assistant message." in user_prompt


@pytest.mark.anyio
@pytest.mark.parametrize("empty_msg", ["", "   ", "\n\t  \n"])
async def test_extract_rejects_empty_or_whitespace_message(empty_msg: str) -> None:
    model = AsyncMock()
    extractor = AppointmentExtractor(chat_model=model)

    with pytest.raises(
        AppointmentExtractionError, match="Empty patient message provided"
    ):
        await extractor.extract(patient_message=empty_msg, current_details={})

    model.ainvoke.assert_not_awaited()


@pytest.mark.anyio
async def test_extract_timeout_raises_appointment_extraction_error() -> None:
    model = AsyncMock()
    model.ainvoke.side_effect = TimeoutError()
    extractor = AppointmentExtractor(chat_model=model)

    with pytest.raises(
        AppointmentExtractionError,
        match="Appointment extraction timeout after 15 seconds",
    ):
        await extractor.extract(
            patient_message="Book tomorrow",
            current_details={},
        )


@pytest.mark.anyio
async def test_extract_model_failure_raises_appointment_extraction_error() -> None:
    model = AsyncMock()
    model.ainvoke.side_effect = RuntimeError("Service unavailable")
    extractor = AppointmentExtractor(chat_model=model)

    with pytest.raises(
        AppointmentExtractionError, match="Failed to extract appointment details"
    ):
        await extractor.extract(
            patient_message="Book tomorrow",
            current_details={},
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "invalid_content",
    [None, 12345, ["chunk1", "chunk2"], {"key": "val"}],
)
async def test_extract_non_string_response_raises_appointment_extraction_error(
    invalid_content: object,
) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = SimpleNamespace(content=invalid_content)
    extractor = AppointmentExtractor(chat_model=model)

    with pytest.raises(
        AppointmentExtractionError, match="Invalid response format from language model"
    ):
        await extractor.extract(
            patient_message="Book tomorrow",
            current_details={},
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "malformed_json",
    ["Not json at all", "{invalid: json}", "```json\n{bad}\n```"],
)
async def test_extract_malformed_json_raises_appointment_extraction_error(
    malformed_json: str,
) -> None:
    model = AsyncMock()
    model.ainvoke.return_value = AIMessage(content=malformed_json)
    extractor = AppointmentExtractor(chat_model=model)

    with pytest.raises(
        AppointmentExtractionError,
        match="Failed to parse appointment details from language model response",
    ):
        await extractor.extract(
            patient_message="Book tomorrow",
            current_details={},
        )


def test_extracted_appointment_details_defaults() -> None:
    details = ExtractedAppointmentDetails()
    assert details.department is None
    assert details.preferred_date is None
    assert details.preferred_time is None
    assert details.contact is None
    assert details.reason is None
    assert details.consent_given is None
