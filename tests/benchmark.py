
import csv
import json
import statistics
import uuid
from collections import defaultdict
from pathlib import Path

from src.graph import graph
from src.observability import AgentObservability


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# Each group represents an independent conversation.
# Questions within the same group share memory.

CASES = [
    # Technical questions
    {
        "group": "technical_1",
        "question": "¿Para qué sirve Spoon en Pentaho 9.3?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
    },
    {
        "group": "technical_2",
        "question": "¿Cómo funciona el licenciamiento de Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
    },
    {
        "group": "technical_3",
        "question": "¿Qué es una transformación en Pentaho 9.3?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
    },
    {
        "group": "technical_4",
        "question": "¿Qué tipos de usuarios existen en Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
    },
    {
        "group": "technical_5",
        "question": "¿Qué información contiene Versions and Releases de Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
    },

    # Conversational memory
    {
        "group": "memory_1",
        "question": "¿Para qué sirve Spoon en Pentaho 9.3?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
    },
    {
        "group": "memory_1",
        "question": "¿Y qué ventajas tiene?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
    },
    {
        "group": "memory_2",
        "question": "¿Cómo funciona el licenciamiento de Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
    },
    {
        "group": "memory_2",
        "question": "¿Y qué documentación tienes sobre esa plataforma?",
        "product": "Pyramid Analytics",
        "intent": "documentation_inventory",
    },

    # Document inventory
    {
        "group": "inventory_1",
        "question": "¿Qué manuales tienes disponibles de Pentaho 9.3?",
        "product": "Pentaho Data Integration",
        "intent": "documentation_inventory",
    },
    {
        "group": "inventory_2",
        "question": "¿Qué documentación tienes sobre Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "documentation_inventory",
    },
    {
        "group": "inventory_3",
        "question": "¿Qué temas están documentados de Pentaho Data Integration?",
        "product": "Pentaho Data Integration",
        "intent": "documentation_inventory",
    },
    {
        "group": "inventory_4",
        "question": "¿Qué documentos tienes indexados de Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "documentation_inventory",
    },

    # Product switching
    {
        "group": "switch_1",
        "question": "¿Qué es Spoon en Pentaho 9.3?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
    },
    {
        "group": "switch_1",
        "question": "Ahora explícame el licenciamiento de Pyramid Analytics.",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
    },
    {
        "group": "switch_1",
        "question": "¿Qué documentación tienes de esa plataforma?",
        "product": "Pyramid Analytics",
        "intent": "documentation_inventory",
    },

    # Questions without sufficient evidence
    {
        "group": "unknown_1",
        "question": "¿Cómo configurar Pentaho 9.3 para conectarse a una base de datos ficticia llamada ZetaDB?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
        "expect_abstention": True,
    },
    {
        "group": "unknown_2",
        "question": "¿Cuál será el precio de Pyramid Analytics en el año 2035?",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
        "expect_abstention": True,
    },
    {
        "group": "unknown_3",
        "question": "¿Cuál es la contraseña de administrador de Pyramid Analytics?",
        "product": "Pyramid Analytics",
        "intent": "technical_question",
        "expect_abstention": True,
    },
    {
        "group": "unknown_4",
        "question": "¿Qué funcionalidades tendrá Pentaho Data Integration en 2040?",
        "product": "Pentaho Data Integration",
        "intent": "technical_question",
        "expect_abstention": True,
    },
]


def percentile(values, p):
    if not values:
        return None

    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)

    return round(
        ordered[low] +
        (ordered[high] - ordered[low]) * (index - low),
        4,
    )


def run_benchmark(limit=None):

    run_id = str(uuid.uuid4())
    sessions = {}
    results = []

    selected = CASES[:limit] if limit else CASES

    for index, case in enumerate(selected, 1):

        group = case["group"]

        if group not in sessions:
            sessions[group] = (
                f"benchmark-{run_id}-{group}"
            )

        observer = AgentObservability()

        try:
            result = graph.invoke(
                {"question": case["question"]},
                config={
                    "configurable": {
                        "thread_id": sessions[group]
                    },
                    "callbacks": [observer],
                },
            )

            record = observer.summary(
                result,
                case["question"],
            )

            validation = record.get("validation") or {}

            expected_strategy = (
                "inventory"
                if case["intent"] == "documentation_inventory"
                else "semantic"
            )

            record.update({
                "case_number": index,
                "group": group,
                "expected_product": case["product"],
                "expected_intent": case["intent"],
                "expected_strategy": expected_strategy,
                "expect_abstention": case.get(
                    "expect_abstention", False
                ),
                "answer": result.get("answer"),
                "routing_correct": (
                    result.get("product") == case["product"]
                    and result.get("intent") == case["intent"]
                ),
                "strategy_correct": (
                    record.get("retrieval_strategy")
                    == expected_strategy
                ),
                "escalated": record.get("escalated"),
                "grounded": validation.get("grounded"),
                "citations_valid": validation.get(
                    "citations_valid"
                ),
                "answers_question": validation.get(
                    "answers_question"
                ),
                # Abstention requires separate human review.
                "abstention_verified": None,
                "error": None,
            })

        except Exception as exc:
            record = {
                "case_number": index,
                "group": group,
                "question": case["question"],
                "expected_product": case["product"],
                "expected_intent": case["intent"],
                "routing_correct": False,
                "strategy_correct": False,
                "error": str(exc),
            }

        results.append(record)

        print(
            f"[{index}/{len(selected)}] "
            f"{case['question']}"
        )
        print(
            "  Routing:",
            record.get("routing_correct"),
            "| Strategy:",
            record.get("strategy_correct"),
            "| Latency:",
            record.get("latency_seconds"),
            "| Error:",
            record.get("error"),
        )

    return results


def create_summary(results):

    successful = [
        r for r in results
        if not r.get("error")
    ]

    latencies = [
        r["latency_seconds"]
        for r in successful
        if r.get("latency_seconds") is not None
    ]

    def rate(field):
        values = [
            r.get(field)
            for r in results
        ]
        return round(
            100 * sum(v is True for v in values)
            / len(values),
            2,
        ) if values else None

    summary = {
        "total_cases": len(results),
        "successful_executions": len(successful),
        "execution_success_rate": round(
            100 * len(successful) / len(results), 2
        ) if results else None,
        "routing_accuracy": rate("routing_correct"),
        "strategy_accuracy": rate("strategy_correct"),
        "validator_grounded_rate": rate("grounded"),
        "validator_citation_rate": rate("citations_valid"),
        "validator_answer_rate": rate("answers_question"),
        "mean_latency_seconds": round(
            statistics.mean(latencies), 4
        ) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "mean_total_tokens": round(
            statistics.mean([
                r.get("total_tokens", 0)
                for r in successful
            ]), 2
        ) if successful else None,
        "escalation_rate": rate("escalated"),
        "revision_rate": round(
            100 * sum(
                (r.get("revision_count") or 0) > 0
                for r in successful
            ) / len(successful), 2
        ) if successful else None,
    }

    return summary


def save_results(results, summary):

    json_path = RESULTS_DIR / "benchmark_results.json"
    csv_path = RESULTS_DIR / "benchmark_summary.csv"

    json_path.write_text(
        json.dumps(
            {
                "results": results,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for key, value in summary.items():
            writer.writerow([key, value])

    print("\nBENCHMARK SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    results = run_benchmark(limit=args.limit)
    summary = create_summary(results)
    save_results(results, summary)
