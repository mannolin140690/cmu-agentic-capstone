
from typing import TypedDict, NotRequired, Any


class AgentState(TypedDict):
    question: str
    

    # Contexto conversacional
    previous_question: NotRequired[str | None]
    previous_answer: NotRequired[str | None]
    product: NotRequired[str | None]
    version: NotRequired[str | None]
    products: NotRequired[list[str]]

    # Decisiones del supervisor
    needs_retrieval: NotRequired[bool]
    needs_clarification: NotRequired[bool]
    clarification_question: NotRequired[str | None]
    routing_reason: NotRequired[str]

    # Ejecución de la consulta actual
    evidence: NotRequired[list[dict[str, Any]]]
    answer: NotRequired[str]
    validation: NotRequired[dict[str, Any] | None]
    final_validation: NotRequired[dict[str, Any] | None]
    revision_count: NotRequired[int]
    metrics: NotRequired[dict[str, Any]]

    #Query Rewritter
    rewritten_query: NotRequired[str]
    last_question: NotRequired[str | None]

    intent: NotRequired[str]

    # Human oversight
    escalated: NotRequired[bool]
    handoff: NotRequired[dict[str, Any] | None]
