import pytest

from guardrails.emergency import EmergencyGuardrail, EmergencyRule
from guardrails.models import SafetyAction


@pytest.fixture
def guardrail() -> EmergencyGuardrail:
    return EmergencyGuardrail(
        rules=(
            EmergencyRule(
                name="test_emergency",
                phrases=("approved test phrase",),
            ),
        )
    )


def test_matching_rule_returns_emergency(guardrail: EmergencyGuardrail) -> None:
    result = guardrail.evaluate("This contains an APPROVED TEST PHRASE.")

    assert result.action == SafetyAction.EMERGENCY
    assert result.reason == "test_emergency"
    assert result.response is not None


def test_safe_message_can_continue(guardrail: EmergencyGuardrail) -> None:
    result = guardrail.evaluate("Where is the hospital pharmacy?")

    assert result.action == SafetyAction.CONTINUE
    assert result.reason is None
    assert result.response is None


def test_rules_are_required() -> None:
    with pytest.raises(ValueError, match="At least one emergency rule is required"):
        EmergencyGuardrail(rules=())
