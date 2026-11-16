from qdrant_client import models

from src.config import (
    qdrant_client,
    embedding_model,
    COLLECTION_NAME
)


def search_documentation(
    question: str,
    product: str,
    version: str | None = None,
    limit: int = 3
) -> list[dict]:

    vector = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    conditions = [
        models.FieldCondition(
            key="product",
            match=models.MatchValue(value=product)
        )
    ]

    if version:
        conditions.append(
            models.FieldCondition(
                key="version",
                match=models.MatchValue(value=version)
            )
        )

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=models.Filter(must=conditions),
        limit=limit,
        with_payload=True
    ).points

    return [
        {
            "source_id": str(point.id),
            "score": round(point.score, 4),
            "product": point.payload.get("product"),
            "version": point.payload.get("version"),
            "title": point.payload.get("title"),
            "url": point.payload.get("url"),
            "content": point.payload.get("content")
        }
        for point in results
    ]