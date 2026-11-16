
from src.graph import graph
from src.observability import AgentObservability


config = {
    "configurable": {
        "thread_id": "observability-test-001"
    }
}

questions = [
    "¿Cómo funciona el licenciamiento de Pyramid Analytics?",
    "¿Qué documentación tienes sobre esa plataforma?"
]


for question in questions:

    observer = AgentObservability()

    result = graph.invoke(
        {"question": question},
        config={
            **config,
            "callbacks": [observer]
        }
    )

    record = observer.summary(
        result,
        question
    )

    observer.save(record)

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("INTENT:", record["intent"])
    print("LATENCY:", record["latency_seconds"])
    print("LLM CALLS:", record["llm_calls"])
    print("INPUT TOKENS:", record["input_tokens"])
    print("OUTPUT TOKENS:", record["output_tokens"])
    print("RETRIEVAL:", record["retrieval_strategy"])
    print("DOCUMENTS:", record["documents_retrieved"])
    print("REVISIONS:", record["revision_count"])
    print("NODE DURATIONS:", record["node_durations"])
