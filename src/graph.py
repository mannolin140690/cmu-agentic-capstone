from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.retrieval import retrieval_node
from src.agents.researcher import researcher_node
from src.agents.validator import validator_node
from src.agents.reviser import reviser_node
from langgraph.checkpoint.memory import InMemorySaver
from src.agents.query_rewriter import query_rewriter_node
from src.agents.escalation import escalation_node
from src.routing import route_validator

def clarification_node(state: AgentState) -> dict:
    return {
        "answer": (
            state.get("clarification_question")
            or "¿Puedes especificar el producto?"
        )
    }


def route_supervisor(state: AgentState) -> str:

    # First: clarification only when truly necessary
    if state.get("needs_clarification", False):
        return "clarification"

    # Documentation inventory always requires retrieval
    if state.get("intent") == "documentation_inventory":
        return "query_rewriter"

    # Technical questions requiring documentation
    if state.get("needs_retrieval", False):
        return "query_rewriter"

    return "clarification"


def build_graph():

    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("clarification", clarification_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("validator", validator_node)
    builder.add_node("reviser", reviser_node)
    builder.add_node("query_rewriter",query_rewriter_node)
    builder.add_node("escalation", escalation_node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "clarification": "clarification",
            "query_rewriter": "query_rewriter"
        }
    )

    builder.add_edge("clarification", END)
    builder.add_edge("retrieval", "researcher")
    builder.add_edge("researcher", "validator")
    
    builder.add_conditional_edges(
        "validator",
        route_validator,
        {
            "end": END,
            "reviser": "reviser",
            "escalation": "escalation"
        }
    )

    builder.add_edge("reviser", "validator")
    builder.add_edge("escalation", END)
    builder.add_edge("query_rewriter","retrieval")

    return builder.compile(
        checkpointer=InMemorySaver()
    )


graph = build_graph()