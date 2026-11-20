"""
Offline tests for validation routing and human escalation.
No OpenAI key, Qdrant or model download required:

    PYTHONPATH=. python tests/test_escalation.py
"""

import os
import tempfile
from pathlib import Path

from src.routing import route_validator, validation_passed
from src.agents import escalation

OK = {
    "grounded": True,
    "citations_valid": True,
    "answers_question": True,
    "unsupported_claims": [],
}
BAD = {**OK, "grounded": False, "unsupported_claims": ["Claim X"]}


def test_validation_passed():
    assert validation_passed(OK)
    assert not validation_passed(BAD)
    assert not validation_passed(None)
    assert not validation_passed({})


def test_routing():
    # First pass
    assert route_validator({"validation": OK}) == "end"
    assert route_validator({"validation": BAD}) == "reviser"
    assert route_validator({"validation": None}) == "reviser"

    # After the single allowed revision
    assert route_validator(
        {"revision_count": 1, "final_validation": OK}
    ) == "end"
    assert route_validator(
        {"revision_count": 1, "final_validation": BAD}
    ) == "escalation"
    assert route_validator({"revision_count": 1}) == "escalation"


def test_escalation_node():
    state = {
        "question": "¿Cómo configuro X en Pentaho 9.3?",
        "product": "Pentaho Data Integration",
        "version": "9.3",
        "answer": "Borrador no verificado",
        "revision_count": 1,
        "final_validation": BAD,
        "evidence": [{"source_id": "1", "title": "Doc", "url": "u"}],
    }

    with tempfile.TemporaryDirectory() as tmp:
        escalation.HANDOFF_PATH = Path(tmp) / "handoffs.jsonl"
        result = escalation.escalation_node(state)
        assert escalation.HANDOFF_PATH.exists()

    assert result["escalated"] is True
    assert result["handoff"]["status"] == "pending_human_review"
    assert result["handoff"]["unverified_draft"] == "Borrador no verificado"
    # The unverified draft must never reach the user as the answer.
    assert "Borrador no verificado" not in result["answer"]
    assert "Claim X" in result["answer"]


if __name__ == "__main__":
    test_validation_passed()
    test_routing()
    test_escalation_node()
    print("ESCALATION TESTS: 3/3 PASSED")
