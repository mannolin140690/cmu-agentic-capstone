
import uuid
from pathlib import Path

import streamlit as st

from src.graph import graph
from src.observability import AgentObservability


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="CMU Agentic AI Capstone",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Technical Documentation Agent")
st.caption(
    "CMU Agentic AI Capstone | "
    "LangGraph + OpenAI + Qdrant + Streamlit"
)


# ---------------------------------------------------------
# Session management
# ---------------------------------------------------------

def initialize_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "traces" not in st.session_state:
        st.session_state.traces = []


initialize_session()


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.traces = []


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def get_sources(evidence):
    """
    Deduplicate retrieved sources.

    Works with semantic retrieval and
    documentation inventory evidence.
    """
    sources = []
    seen = set()

    for item in evidence or []:
        if not isinstance(item, dict):
            continue

        source_id = item.get("source_id")
        title = item.get("title") or "Untitled document"
        url = item.get("url")
        product = item.get("product")
        version = item.get("version")

        key = (source_id, title, url)

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source_id": source_id,
            "title": title,
            "url": url,
            "product": product,
            "version": version,
            "chunks": item.get("chunks"),
        })

    return sources


def render_sources(sources):
    if not sources:
        st.caption("No document sources available.")
        return

    for index, source in enumerate(sources, 1):
        st.markdown(
            f"**{index}. {source['title']}**"
        )

        details = []

        if source.get("product"):
            details.append(source["product"])

        if source.get("version"):
            details.append(
                f"Version: {source['version']}"
            )

        if source.get("chunks") is not None:
            details.append(
                f"Chunks: {source['chunks']}"
            )

        if details:
            st.caption(" | ".join(details))

        if source.get("url"):
            st.link_button(
                "Open documentation",
                source["url"],
            )

        if source.get("source_id"):
            st.code(
                str(source["source_id"]),
                language=None,
            )


def render_trace(trace):
    if not trace:
        return

    st.markdown("#### Execution metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Latency",
        f"{trace.get('latency_seconds', 0):.2f} s",
    )
    c2.metric(
        "LLM calls",
        trace.get("llm_calls", 0),
    )
    c3.metric(
        "Total tokens",
        trace.get("total_tokens", 0),
    )
    c4.metric(
        "Revisions",
        trace.get("revision_count", 0),
    )

    st.markdown("#### Agent decisions")

    st.write(
        "**Product:**",
        trace.get("product") or "Not identified",
    )
    st.write(
        "**Intent:**",
        trace.get("intent") or "Unknown",
    )
    st.write(
        "**Retrieval strategy:**",
        trace.get("retrieval_strategy") or "None",
    )
    st.write(
        "**Documents retrieved:**",
        trace.get("documents_retrieved", 0),
    )

    validation = trace.get("validation") or {}

    st.markdown("#### Validation")

    st.write(
        "**Grounded:**",
        validation.get("grounded"),
    )
    st.write(
        "**Citations valid:**",
        validation.get("citations_valid"),
    )
    st.write(
        "**Answers question:**",
        validation.get("answers_question"),
    )

    if validation.get("feedback"):
        st.caption(validation["feedback"])

    with st.expander("Node execution times"):
        st.json(trace.get("node_durations") or {})

    with st.expander("Full trace"):
        st.json(trace)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Agent configuration")

    st.markdown("**Supported products**")
    st.write("• Pentaho Data Integration 9.3")
    st.write("• Pyramid Analytics")

    st.divider()

    st.markdown("**Architecture**")
    st.write("Supervisor")
    st.write("Query Rewriter")
    st.write("Retrieval")
    st.write("Researcher")
    st.write("Validator")
    st.write("Reviser")

    st.divider()

    if st.button(
        "➕ New conversation",
        use_container_width=True,
    ):
        new_conversation()
        st.rerun()

    st.caption(
        f"Session: {st.session_state.thread_id[:8]}"
    )

    st.caption(
        f"Messages: {len(st.session_state.messages)}"
    )


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources") or []
            trace = message.get("trace")

            with st.expander(
                f"📚 Sources ({len(sources)})"
            ):
                render_sources(sources)

            with st.expander("📊 Observability"):
                render_trace(trace)


# ---------------------------------------------------------
# Chat input and agent execution
# ---------------------------------------------------------

question = st.chat_input(
    "Ask about Pentaho or Pyramid Analytics..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agents are working..."):
            observer = AgentObservability()

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id,
                },
                "callbacks": [observer],
            }

            try:
                result = graph.invoke(
                    {"question": question},
                    config=config,
                )

                answer = (
                    result.get("answer")
                    or result.get("clarification_question")
                    or "No response was generated."
                )

                sources = get_sources(
                    result.get("evidence") or []
                )

                trace = observer.summary(
                    result,
                    question,
                )

                observer.save(trace)

                st.markdown(answer)

                with st.expander(
                    f"📚 Sources ({len(sources)})"
                ):
                    render_sources(sources)

                with st.expander("📊 Observability"):
                    render_trace(trace)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "trace": trace,
                })

                st.session_state.traces.append(trace)

            except Exception as exc:
                st.error(
                    "The agent could not complete the request."
                )
                st.exception(exc)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "CMU Agentic AI Capstone | "
    "Multi-Agent RAG System | "
    "LangGraph + Qdrant + OpenAI"
)
