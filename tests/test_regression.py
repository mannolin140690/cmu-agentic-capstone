
import uuid

from src.graph import graph


RUN_ID = str(uuid.uuid4())

TESTS = [
    {
        "question": (
            "¿Qué información contiene Versions and "
            "Releases de Pyramid Analytics?"
        ),
        "intent": "technical_question",
        "product": "Pyramid Analytics",
        "strategy": "semantic",
        "session": "document_content",
    },
    {
        "question": (
            "¿Qué documentación tienes sobre "
            "Pyramid Analytics?"
        ),
        "intent": "documentation_inventory",
        "product": "Pyramid Analytics",
        "strategy": "inventory",
        "session": "inventory_pyramid",
    },
    {
        "question": (
            "¿Qué manuales tienes disponibles "
            "de Pentaho 9.3?"
        ),
        "intent": "documentation_inventory",
        "product": "Pentaho Data Integration",
        "strategy": "inventory",
        "session": "inventory_pentaho",
    },
    {
        "question": (
            "¿Cómo funciona el licenciamiento "
            "de Pyramid Analytics?"
        ),
        "intent": "technical_question",
        "product": "Pyramid Analytics",
        "strategy": "semantic",
        "session": "conversation",
    },
    {
        "question": (
            "¿Y qué documentación tienes "
            "sobre esa plataforma?"
        ),
        "intent": "documentation_inventory",
        "product": "Pyramid Analytics",
        "strategy": "inventory",
        "session": "conversation",
    },
    {
        "question": (
            "¿Qué información contiene el "
            "documento Versions and Releases?"
        ),
        "intent": "technical_question",
        "product": "Pyramid Analytics",
        "strategy": "semantic",
        "session": "conversation",
    },
]


def main():
    passed = 0

    for index, test in enumerate(TESTS, 1):
        config = {
            "configurable": {
                "thread_id": (
                    f"regression-{RUN_ID}-"
                    f"{test['session']}"
                )
            }
        }

        try:
            result = graph.invoke(
                {"question": test["question"]},
                config=config,
            )

            metrics = result.get("metrics") or {}

            actual = {
                "intent": result.get("intent"),
                "product": result.get("product"),
                "strategy": metrics.get(
                    "retrieval_strategy"
                ),
            }

            expected = {
                "intent": test["intent"],
                "product": test["product"],
                "strategy": test["strategy"],
            }

            success = actual == expected

            if success:
                passed += 1

            print("\n" + "=" * 60)
            print(f"TEST {index}: {'PASS' if success else 'FAIL'}")
            print("QUESTION:", test["question"])
            print("EXPECTED:", expected)
            print("ACTUAL:", actual)
            print(
                "SOURCES:",
                len(result.get("evidence") or [])
            )
            print(
                "REVISIONS:",
                result.get("revision_count", 0)
            )

            if not success:
                print(
                    "ROUTING REASON:",
                    result.get("routing_reason")
                )

        except Exception as exc:
            print("\n" + "=" * 60)
            print(f"TEST {index}: ERROR")
            print("ERROR:", repr(exc))

    print("\n" + "=" * 60)
    print(f"REGRESSION RESULT: {passed}/{len(TESTS)} PASSED")


if __name__ == "__main__":
    main()
