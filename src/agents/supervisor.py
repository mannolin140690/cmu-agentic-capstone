
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME


SUPPORTED_PRODUCTS = [
    "Pentaho Data Integration",
    "Pyramid Analytics"
]


class RoutingDecision(BaseModel):

    product: str | None = Field(
        description="Primary product, or null"
    )

    products: list[str] = Field(
        default_factory=list,
        description="All products involved in the question"
    )

    version: str | None = Field(
        description="Requested or inherited product version"
    )

    intent: Literal[
        "technical_question",
        "documentation_inventory"
    ] = Field(
        description="Current user intent"
    )

    needs_retrieval: bool = Field(
        description="Whether documentation retrieval is required"
    )

    needs_clarification: bool = Field(
        description="Whether clarification is required"
    )

    clarification_question: str | None = Field(
        description="Clarification question, or null"
    )

    reasoning: str = Field(
        description="Reason for the routing decision"
    )


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
)

structured_llm = llm.with_structured_output(
    RoutingDecision
)


def supervisor_node(state: dict) -> dict:

    question = state["question"]

    # Previous conversation context
    previous_product = state.get("product")
    previous_version = state.get("version")
    previous_question = state.get("last_question")
    previous_answer = state.get("answer")

    decision = structured_llm.invoke([
        (
            "system",
            """
You are the contextual supervisor of a technical
multi-agent documentation assistant.

SUPPORTED PRODUCTS:
- Pentaho Data Integration
- Pyramid Analytics

PRODUCT AND CONTEXT RULES:

1. Spoon and PDI refer to Pentaho Data Integration.

2. Identify all explicitly mentioned products.

3. If the question is a follow-up and does not
   specify a product, inherit the previous product
   only when context clearly supports it.

4. Inherit the previous version only when the
   question continues discussing the same product.

5. An explicitly requested version always
   overrides the previous version.

6. Never invent product versions.

7. If the question switches products, do not
   carry over the previous product's version.

8. If the product cannot be determined from
   the question or context, ask for clarification.

9. Use only supported product names.

10. When multiple products are involved,
    include all of them in the products list.

INTENT CLASSIFICATION RULES:

11. Classify the CURRENT user question,
    independently of the previous intent.

12. Use technical_question for questions about:
    - Product functionality
    - Technical concepts
    - Configuration procedures
    - Troubleshooting
    - Product features
    - Advantages or disadvantages
    - Specific document content

13. Use documentation_inventory for requests about:
    - Available documentation
    - Indexed documents
    - Available manuals or guides
    - Document catalogs
    - Indexed topics
    - Available documented versions

14. Examples of documentation_inventory:
    - What documentation is available?
    - What documents do you have?
    - Which manuals are indexed?
    - What topics are documented?
    - ¿Qué documentación tienes?
    - ¿Qué manuales tienes disponibles?
    - ¿Qué documentación tienes sobre esa plataforma?
    - ¿Qué versiones tienes documentadas?

15. A follow-up may preserve the product
    and version while changing its intent.

16. Documentation inventory ALWAYS requires
    retrieval from the document database.

17. If the product is already identified,
    do not ask for clarification merely because
    the user requests a documentation inventory.

18. Technical questions require documentation
    retrieval whenever the product is identified.

EXAMPLE:

Previous question:
¿Cómo funciona el licenciamiento de Pyramid Analytics?

Current question:
¿Y qué documentación tienes sobre esa plataforma?

Expected routing:
product = Pyramid Analytics
intent = documentation_inventory
needs_retrieval = true
needs_clarification = false

Return a structured routing decision.

DOCUMENT CONTENT VS DOCUMENT INVENTORY:

19. A question about the CONTENT of a specific
    named document is a technical_question.

20. A documentation_inventory request asks which
    documents, manuals, guides, or indexed topics
    are AVAILABLE.

21. Do not classify a question as
    documentation_inventory merely because it
    contains the words "document", "documentation",
    "manual", or a document title.

EXAMPLES:

Question:
"What information does Versions and Releases contain?"
intent = technical_question

Question:
"¿Qué información contiene el documento
Versions and Releases de Pyramid Analytics?"
intent = technical_question

Question:
"¿Qué documentos de Pyramid Analytics tienes?"
intent = documentation_inventory

Question:
"¿Qué manuales están indexados?"
intent = documentation_inventory
"""
        ),
        (
            "human",
            f"""
CURRENT QUESTION:
{question}

PREVIOUS PRODUCT:
{previous_product}

PREVIOUS VERSION:
{previous_version}

PREVIOUS QUESTION:
{previous_question}

PREVIOUS ANSWER:
{previous_answer}
"""
        )
    ])

    # Validate supported products
    products = [
        p for p in decision.products
        if p in SUPPORTED_PRODUCTS
    ]

    if (
        decision.product in SUPPORTED_PRODUCTS
        and decision.product not in products
    ):
        products.insert(0, decision.product)

    primary = (
        decision.product
        if decision.product in SUPPORTED_PRODUCTS
        else None
    )

    # Recover primary product if the model returned
    # a valid products list but omitted the primary.
    if primary is None and products:
        primary = products[0]

    # Normalize routing flags
    needs_retrieval = decision.needs_retrieval
    needs_clarification = decision.needs_clarification
    clarification_question = decision.clarification_question

    # Deterministic guardrail:
    # Inventory requests with identified products
    # must always retrieve documentation.
    if (
        decision.intent == "documentation_inventory"
        and products
    ):
        needs_retrieval = True
        needs_clarification = False
        clarification_question = None

    # Technical questions with identified products
    # must also retrieve documentation.
    elif (
        decision.intent == "technical_question"
        and products
    ):
        needs_retrieval = True
        needs_clarification = False
        clarification_question = None

    # No supported product identified.
    elif not products:
        needs_retrieval = False
        needs_clarification = True

        if not clarification_question:
            clarification_question = (
                "¿Tu pregunta se refiere a Pentaho "
                "Data Integration o Pyramid Analytics?"
            )

    return {
        # Conversation memory
        "previous_question": previous_question,
        "previous_answer": previous_answer,
        "last_question": question,

        # Product context
        "product": primary,
        "products": products,
        "version": decision.version,

        # Intent and routing
        "intent": decision.intent,
        "needs_retrieval": needs_retrieval,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "routing_reason": decision.reasoning,

        # Reset current-turn results
        "rewritten_query": "",
        "evidence": [],
        "answer": "",
        "validation": None,
        "final_validation": None,
        "revision_count": 0,
        "escalated": False,
        "handoff": None,
        "metrics": {}
    }
