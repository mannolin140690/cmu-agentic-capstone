"""Pure routing helpers (no LLM or database imports, so they are unit-testable offline)."""


def validation_passed(result: dict | None) -> bool:
    """True when the validator approved the answer on every criterion."""
    if not result:
        return False

    return bool(
        result.get("grounded", False)
        and result.get("citations_valid", False)
        and result.get("answers_question", False)
        and not result.get("unsupported_claims", [])
    )


def route_validator(state: dict) -> str:
    """
    Decide what happens after validation.

    - First pass approved                 -> end
    - First pass rejected                 -> reviser (one bounded revision)
    - After the revision, approved        -> end
    - After the revision, still rejected  -> escalation (human review)
    """
    if state.get("revision_count", 0) >= 1:
        if validation_passed(state.get("final_validation")):
            return "end"
        return "escalation"

    if validation_passed(state.get("validation")):
        return "end"

    return "reviser"
