
import json
from pathlib import Path

from qdrant_client import QdrantClient

COLLECTION = "cmu_technical_docs"

client = QdrantClient(url="http://localhost:6333")

dataset = json.loads(
    Path("tests/golden_questions_v1.json").read_text(
        encoding="utf-8"
    )
)

for case in dataset["cases"]:
    print("\n" + "=" * 70)
    print(f"{case['id']}: {case['question']}")

    print("\nEXPECTED FACTS:")
    for fact in case["expected_facts"]:
        print("-", fact)

    print("\nORIGINAL EVIDENCE:")

    for source_id in case["reference_source_ids"]:
        points = client.retrieve(
            collection_name=COLLECTION,
            ids=[source_id],
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            print("SOURCE NOT FOUND:", source_id)
            continue

        for point in points:
            payload = point.payload or {}

            print("\nSOURCE ID:", point.id)
            print("TITLE:", payload.get("title"))
            print(
                "CONTENT:",
                payload.get("content")
                or payload.get("text")
                or "No text field found",
            )
