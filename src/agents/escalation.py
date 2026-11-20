import json
from datetime import datetime, timezone
from pathlib import Path

HANDOFF_PATH = Path("logs/handoffs.jsonl")


def build_handoff(state: dict) -> dict:
    """Structured package for the platform administration team."""
    validation = state.get("final_validation") or {}

    return {
        "status": "pending_human_review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Answer could not be verified against the retrieved "
            "documentation after one revision."
        ),
        "question": state.get("question"),
        "rewritten_query": state.get("rewritten_query"),
        "product": state.get("product"),
        "version": state.get("version"),
        "intent": state.get("intent"),
        "revision_count": state.get("revision_count", 0),
        "validation": validation,
        "evidence": [
            {
                "source_id": item.get("source_id"),
                "title": item.get("title"),
                "version": item.get("version"),
                "url": item.get("url"),
                "score": item.get("score"),
            }
            for item in (state.get("evidence") or [])
        ],
        # Kept for the reviewer only. It is NOT shown to the user as verified.
        "unverified_draft": state.get("answer"),
    }


def escalation_node(state: dict) -> dict:
    """
    Replace an unverified answer with a limitation notice and
    simulate the human handoff by logging a structured package.
    """
    handoff = build_handoff(state)

    try:
        HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HANDOFF_PATH.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(handoff, ensure_ascii=False, default=str)
                + "\n"
            )
    except OSError:
        pass  # Logging is best-effort; escalation must not fail.

    unsupported = (
        (state.get("final_validation") or {})
        .get("unsupported_claims", [])
    )[:3]

    lines = [
        "⚠️ No pude verificar esta respuesta contra la documentación "
        "indexada, así que no la presento como confirmada. "
        "El caso quedó registrado para revisión del equipo de "
        "administración de la plataforma.",
    ]

    if unsupported:
        lines.append("\nAfirmaciones que no se pudieron respaldar:")
        lines.extend(f"- {claim}" for claim in unsupported)

    lines.append(
        "\nPuedes reformular la pregunta indicando producto y versión, "
        "o consultar directamente la documentación oficial."
    )

    return {
        "answer": "\n".join(lines),
        "handoff": handoff,
        "escalated": True,
    }
