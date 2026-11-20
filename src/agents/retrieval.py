
from collections import defaultdict
from typing import Any

from qdrant_client import models

from src.config import qdrant_client, COLLECTION_NAME
from src.tools.qdrant_search import search_documentation


PAGE_SIZE = 100


def retrieve_technical_documents(
    state: dict
) -> list[dict[str, Any]]:
    """
    Semantic retrieval for technical questions.
    Uses the rewritten query when available.
    """

    question = (
        state.get("rewritten_query")
        or state["question"]
    )

    products = (
        state.get("products")
        or [state.get("product")]
    )

    primary_product = state.get("product")
    version = state.get("version")

    evidence = []

    for product in dict.fromkeys(products):

        if not product:
            continue

        product_version = (
            version
            if product == primary_product
            else None
        )

        documents = search_documentation(
            question=question,
            product=product,
            version=product_version,
            limit=3
        )

        evidence.extend(documents)

    return evidence


def retrieve_document_inventory(
    state: dict
) -> list[dict[str, Any]]:
    """
    Enumerate indexed documents using Qdrant scroll.

    Unlike semantic retrieval, this reads all matching
    points and groups chunks into documents.
    """

    products = (
        state.get("products")
        or [state.get("product")]
    )

    primary_product = state.get("product")
    version = state.get("version")

    inventory = []

    for product in dict.fromkeys(products):

        if not product:
            continue

        conditions = [
            models.FieldCondition(
                key="product",
                match=models.MatchValue(
                    value=product
                )
            )
        ]

        if version and product == primary_product:
            conditions.append(
                models.FieldCondition(
                    key="version",
                    match=models.MatchValue(
                        value=version
                    )
                )
            )

        query_filter = models.Filter(
            must=conditions
        )

        offset = None

        documents = defaultdict(
            lambda: {
                "source_ids": [],
                "chunks": 0,
                "sample_content": ""
            }
        )

        while True:

            points, next_offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            for point in points:

                payload = point.payload or {}

                title = (
                    payload.get("title")
                    or "Untitled document"
                )

                doc_version = payload.get("version")
                url = payload.get("url")

                key = (
                    product,
                    doc_version,
                    title,
                    url
                )

                document = documents[key]

                document["source_ids"].append(
                    str(point.id)
                )

                document["chunks"] += 1

                if not document["sample_content"]:
                    document["sample_content"] = (
                        payload.get("content") or ""
                    )[:500]

            if next_offset is None:
                break

            offset = next_offset

        for key, data in documents.items():

            (
                doc_product,
                doc_version,
                title,
                url
            ) = key

            inventory.append({
                "source_id": data["source_ids"][0],
                "source_ids": data["source_ids"],
                "product": doc_product,
                "version": doc_version,
                "title": title,
                "url": url,
                "chunks": data["chunks"],
                "content": data["sample_content"],
                "retrieval_type": "inventory"
            })

    return sorted(
        inventory,
        key=lambda item: (
            item["product"] or "",
            item["title"] or ""
        )
    )


def retrieval_node(state: dict) -> dict:
    """
    Select retrieval strategy based on user intent.
    """

    intent = state.get(
        "intent",
        "technical_question"
    )

    # Strategy 1: Documentation inventory
    if intent == "documentation_inventory":

        evidence = retrieve_document_inventory(
            state
        )

        return {
            "evidence": evidence,
            "metrics": {
                "retrieval_strategy": "inventory",
                "documents_retrieved": len(evidence),
                "inventory_complete": True
            }
        }

    # Strategy 2: Semantic search
    evidence = retrieve_technical_documents(
        state
    )

    return {
        "evidence": evidence,
        "metrics": {
            "retrieval_strategy": "semantic",
            "documents_retrieved": len(evidence),
            "inventory_complete": False
        }
    }
