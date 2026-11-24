from agent.workflows.medical_agent.state import (
    answer_result_from_state,
    answer_result_to_state,
)
from services.answering import SAFETY_NOTICE, AnswerResult
from services.evidence import Evidence


def test_answer_result_round_trip_uses_checkpoint_safe_values() -> None:
    result = AnswerResult(
        answer="Fever is an elevated body temperature [C1].",
        citations=[
            Evidence(
                citation_id="C1",
                chunk_id="chunk-1",
                text="Fever is an elevated body temperature.",
                document_id="web",
                page_number=0,
                source="https://medlineplus.gov/fever.html",
                score=0.91,
                title="Fever",
            )
        ],
        sufficient_evidence=True,
        safety_notice=SAFETY_NOTICE,
        route="trusted_web",
    )

    state = answer_result_to_state(result)
    restored = answer_result_from_state(state)

    assert restored == result
    assert isinstance(state["citations"], list)
    assert isinstance(state["citations"][0], dict)


def test_answer_result_from_state_requires_an_answer() -> None:
    try:
        answer_result_from_state({})
    except ValueError as error:
        assert str(error) == "Medical-agent graph completed without an answer."
    else:
        raise AssertionError("Expected missing answer to fail.")
