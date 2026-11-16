from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import MODEL_NAME


class RewrittenQuery(BaseModel):
    query: str = Field(
        description="Standalone technical search query"
    )


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0
).with_structured_output(RewrittenQuery)


def query_rewriter_node(state: dict) -> dict:

    result = llm.invoke([
        (
            "system",
            """
You rewrite conversational technical questions
into standalone queries for document retrieval.

Rules:
- Preserve the user's original intent.
- Include the resolved product and version.
- Resolve pronouns using conversation context.
- Do not invent technical details or procedures.
- Do not assume an integration exists.
- Do not change the requested software version.
- If multiple products are involved, include all.
- Preserve relevant technical entities from the previous
  question, such as Spoon, when resolving references.
- Prefer the language of the indexed documentation
  when it improves retrieval.
- Do not introduce a specific implementation method
  such as JDBC unless the user explicitly requests it
  or the context clearly establishes it.
- Return only the rewritten query.
"""
        ),
        (
            "human",
            f"""
Current question:
{state["question"]}

Resolved products:
{state.get("products")}

Resolved version:
{state.get("version")}

Previous question:
{state.get("previous_question")}
"""
        )
    ])

    return {
        "rewritten_query": result.query
    }