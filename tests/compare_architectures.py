"""Compare direct LLM, deterministic single-pass RAG, and the existing LangGraph agent.

Run from repository root:
    PYTHONPATH=. python tests/compare_architectures.py
    PYTHONPATH=. python tests/compare_architectures.py --limit 2

Outputs raw answers and operational metrics. Manual evaluation must be independent.
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.graph import graph
from src.observability import AgentObservability

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILE = ROOT / "tests" / "golden_questions_v1.json"
OUTPUT_DIR = ROOT / "results"
COLLECTION = os.getenv("COLLECTION_NAME", "cmu_technical_docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def load_cases():
    dataset = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
    if not isinstance(dataset.get("cases"), list):
        raise ValueError("Golden dataset must contain a 'cases' list")
    return dataset["cases"]


def payload_text(payload):
    return str(payload.get("content") or payload.get("text") or "")


def collect_inventory(client, product, version):
    """Scroll through Qdrant, group all points into distinct indexed documents."""
    offset = None
    groups = {}
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for pt in points:
            p = pt.payload or {}
            if str(p.get("product", "")).casefold() != product.casefold():
                continue
            if version and p.get("version") not in (None, "", version):
                continue
            key = (p.get("title"), p.get("url"), p.get("version"))
            item = groups.setdefault(key, {
                "title": p.get("title"), "url": p.get("url"),
                "version": p.get("version"), "source_ids": [], "snippets": [],
            })
            item["source_ids"].append(str(pt.id))
            snippet = payload_text(p).strip()
            if snippet and len(item["snippets"]) < 2:
                item["snippets"].append(snippet[:380])
        if offset is None:
            break
    return list(groups.values())


def retrieve_semantic(client, encoder, question, product, version, top_k=3):
    vector = encoder.encode(question).tolist()
    # Search more broadly first and filter on metadata locally so this works
    # even when the collection has no payload indexes.
    res = client.query_points(
        collection_name=COLLECTION, query=vector,
        limit=max(30, top_k * 10), with_payload=True,
    )
    items = []
    for pt in res.points:
        p = pt.payload or {}
        if str(p.get("product", "")).casefold() != product.casefold():
            continue
        if version and p.get("version") not in (None, "", version):
            continue
        items.append({
            "title": p.get("title"), "url": p.get("url"),
            "version": p.get("version"), "source_ids": [str(pt.id)],
            "content": payload_text(p)[:1600], "score": pt.score,
        })
        if len(items) == top_k:
            break
    return items


def baseline_rag(client, encoder, llm, case):
    question = case["question"]
    product = case["product"]
    version = case.get("version")
    # Fixed, transparent retrieval rule; DOES NOT read the expected answer.
    is_inventory = any(term in question.casefold() for term in (
        "documentos indexados", "documentos de", "documentación disponible",
        "manuales disponibles", "cuáles son los cuatro documentos",
        "cuáles son los tres documentos", "qué documentación tienes",
    ))
    t0 = time.perf_counter()
    if is_inventory:
        docs = collect_inventory(client, product, version)
        evidence = [
            {k: doc[k] for k in ("title", "url", "version", "source_ids")}
            | {"content": "\n".join(doc["snippets"])[:800]}
            for doc in docs
        ]
        strategy = "inventory"
    else:
        evidence = retrieve_semantic(client, encoder, question, product, version)
        strategy = "semantic"
    retrieval_s = round(time.perf_counter() - t0, 4)
    context = "\n\n".join(
        f"Source IDs: {', '.join(d['source_ids'])}\nTitle: {d.get('title')}\n"
        f"URL: {d.get('url')}\nContent:\n{d.get('content', '')}"
        for d in evidence
    )
    messages = [
        ("system", "You answer technical questions using ONLY the supplied documentation. "
         "If the evidence does not answer the question, explicitly say so. "
         "Do not invent details. Cite source IDs for supported technical claims. "
         "Respond in Spanish."),
        ("human", f"Question: {question}\n\nDocumentation evidence:\n{context or '[NONE]'}"),
    ]
    result = llm.invoke(messages)
    usage = result.usage_metadata or {}
    return {
        "answer": str(result.content), "strategy": strategy,
        "source_ids": [sid for d in evidence for sid in d["source_ids"]],
        "documents_retrieved": len(evidence), "retrieval_seconds": retrieval_s,
        "llm_calls": 1, "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def direct_llm(llm, case):
    response = llm.invoke([
        ("system", "Answer technical questions accurately in Spanish. "
         "Do not claim to have accessed proprietary, local, or indexed documents. "
         "Clearly state uncertainty when required."),
        ("human", case["question"]),
    ])
    usage = response.usage_metadata or {}
    return {
        "answer": str(response.content), "source_ids": [], "llm_calls": 1,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def multi_agent(case):
    observer = AgentObservability()
    result = graph.invoke(
        {"question": case["question"]},
        config={"configurable": {"thread_id": f"golden-{uuid.uuid4()}"},
                "callbacks": [observer]},
    )
    trace = observer.summary(result, case["question"])
    evidence = result.get("evidence") or []
    return {
        "answer": result.get("answer") or result.get("clarification_question") or "",
        "source_ids": [str(x.get("source_id")) for x in evidence
                       if isinstance(x, dict) and x.get("source_id")],
        "strategy": trace.get("retrieval_strategy"),
        "documents_retrieved": trace.get("documents_retrieved"),
        "llm_calls": trace.get("llm_calls"),
        "input_tokens": trace.get("input_tokens"),
        "output_tokens": trace.get("output_tokens"),
        "total_tokens": trace.get("total_tokens"),
        "revision_count": trace.get("revision_count"),
        "escalated": trace.get("escalated", result.get("escalated", False)),
        "validation": trace.get("validation"),
        "node_durations": trace.get("node_durations"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run the first N questions (smoke test).")
    args = parser.parse_args()
    cases = load_cases()[:args.limit] if args.limit else load_cases()
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "architecture_comparison.json"
    run = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_FILE.relative_to(ROOT)),
        "model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "methodology": "Independent questions; direct LLM / single-pass RAG / multiagent; manual scoring pending",
        "results": [],
    }
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    client = QdrantClient(url=QDRANT_URL)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['id']} {case['question']}", flush=True)
        for label, fn in (
            ("llm_direct", lambda: direct_llm(llm, case)),
            ("rag_simple", lambda: baseline_rag(client, encoder, llm, case)),
            ("rag_multiagent", lambda: multi_agent(case)),
        ):
            t0 = time.perf_counter()
            try:
                details = fn()
                error = None
            except Exception as exc:
                details = {}
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "case_id": case["id"], "question": case["question"],
                "product": case["product"], "architecture": label,
                "latency_seconds": round(time.perf_counter()-t0, 4),
                **details, "error": error,
                "manual_score": None, "manual_citation_check": None,
                "manual_notes": "",
            }
            run["results"].append(record)
            print(f"  {label}: {record['latency_seconds']} s | "
                  f"tokens={record.get('total_tokens')} | error={error}", flush=True)
            # Write progressively; interruptions do not erase completed cases.
            output_path.write_text(json.dumps(run, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"\nSaved: {output_path}")
    print("Manual evaluation pending: do not interpret these outputs as accuracy scores.")


if __name__ == "__main__":
    main()
