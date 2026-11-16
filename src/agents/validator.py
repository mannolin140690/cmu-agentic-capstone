
import json

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.config import MODEL_NAME


class ValidationResult(BaseModel):

    grounded: bool = Field(
        description=(
            "Whether all factual claims are supported "
            "by the provided evidence"
        )
    )

    citations_valid: bool = Field(
        description=(
            "Whether citations refer to valid source IDs "
            "and support their associated claims"
        )
    )

    answers_question: bool = Field(
        description=(
            "Whether the answer adequately addresses "
            "the user's question, including an honest "
            "abstention when evidence is insufficient"
        )
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Claims that are not supported by evidence"
        )
    )

    feedback: str = Field(
        description=(
            "Specific feedback for correcting the answer"
        )
    )


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
)

structured_llm = llm.with_structured_output(
    ValidationResult
)


def validator_node(state: dict) -> dict:

    question = state["question"]
    rewritten_query = state.get(
        "rewritten_query",
        question
    )

    answer = state.get("answer", "")
    evidence = state.get("evidence", [])

    intent = state.get(
        "intent",
        "technical_question"
    )

    revision_count = state.get(
        "revision_count",
        0
    )

    validation = structured_llm.invoke([
        (
            "system",
            """
You are a strict evidence-grounding validator
for a technical multi-agent RAG system.

Your responsibility is to evaluate whether
the generated answer is supported by the
retrieved documentation.

VALIDATION RULES:

1. Every factual technical claim must be
   supported by the provided evidence.

2. Citations must reference source IDs
   present in the evidence.

3. A citation is not sufficient by itself.
   The cited document must actually support
   the associated claim.

4. Do not infer that documentation does not
   exist merely because it was not retrieved
   among the top-k semantic search results.

5. Distinguish between:
   - Not found in retrieved evidence.
   - Not present in the indexed collection.

6. Collection-wide absence claims require
   an exhaustive inventory or equivalent
   authoritative metadata.

7. Do not confuse related technologies
   with equivalent procedures.

   Example:
   An Oracle Bulk Loader transformation step
   does not establish how to configure a
   generic Oracle JDBC connection.

8. Do not invent configuration parameters,
   APIs, product features or version support.

9. If evidence is insufficient, a cautious
   abstention is acceptable.

10. When the intent is documentation_inventory,
    validate that the response describes
    indexed documents rather than treating
    semantic search results as a full catalog.

11. If the answer contains unsupported claims,
    mark grounded as false and identify them.

12. If a citation does not support its claim,
    mark citations_valid as false.

13. If the response does not adequately answer
    the question, mark answers_question as false.

14. Return structured validation results.
"""
        ),
        (
            "human",
            f"""
ORIGINAL QUESTION:
{question}

REWRITTEN QUERY:
{rewritten_query}

INTENT:
{intent}

GENERATED ANSWER:
{answer}

RETRIEVED EVIDENCE:
{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2
)}
"""
        )
    ])

    result = validation.model_dump()

    if revision_count == 0:
        return {
            "validation": result
        }

    return {
        "final_validation": result
    }
