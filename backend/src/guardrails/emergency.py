from dataclasses import dataclass

from guardrails.models import SafetyAction, SafetyDecision

EMERGENCY_RESPONSE = (
    "This may require immediate medical attention. "
    "Contact your local emergency service or go to the nearest "
    "emergency department now. Do not wait for a response from this assistant."
)


@dataclass(frozen=True)
class EmergencyRule:
    name: str
    phrases: tuple[str, ...]


DEFAULT_EMERGENCY_RULES = (
    EmergencyRule(
        name="explicit_emergency_request",
        phrases=(
            "call an ambulance",
            "medical emergency",
            "need emergency help",
        ),
    ),
)


class EmergencyGuardrail:
    def __init__(self, rules: tuple[EmergencyRule, ...]) -> None:
        if not rules:
            raise ValueError("At least one emergency rule is required.")

        self._rules = tuple(
            EmergencyRule(
                name=rule.name,
                phrases=tuple(phrase.casefold() for phrase in rule.phrases),
            )
            for rule in rules
        )

    def evaluate(self, message: str) -> SafetyDecision:
        normalized_message = message.casefold()

        for rule in self._rules:
            if any(phrase in normalized_message for phrase in rule.phrases):
                return SafetyDecision(
                    action=SafetyAction.EMERGENCY,
                    reason=rule.name,
                    response=EMERGENCY_RESPONSE,
                )

        return SafetyDecision(action=SafetyAction.CONTINUE)
