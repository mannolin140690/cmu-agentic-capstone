# Agentic Analytics Support Assistant

Capstone Project — Carnegie Mellon University, Agentic AI Program

## Overview

A multi-agent assistant for answering technical questions about **Pentaho Data Integration 9.3** and **Pyramid Analytics** using retrieved documentation. Built with Python and LangGraph, it combines conversational memory, RAG, answer validation, and basic observability.

## Architecture

- **Interface:** Streamlit
- **Agents:** Supervisor, Query Rewriter, Retrieval, Researcher, Validator, Reviser
- **LLM:** OpenAI
- **Retrieval:** Qdrant vector database with semantic search and documentation inventory
- **Memory:** LangGraph short-term conversational memory
- **Infrastructure:** Docker Compose for Qdrant

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in your local `.env` file and start Qdrant using the project's Docker Compose configuration. Ensure the documentation collection has been populated before running the app.

```bash
PYTHONPATH=. python -m streamlit run app/streamlit_app.py
```

## Testing

```bash
PYTHONPATH=. python tests/benchmark.py
PYTHONPATH=. python tests/test_regression.py
```

Initial results: **20/20 benchmark executions succeeded**, with **95% routing and retrieval-strategy accuracy**; the subsequent **6/6 regression tests passed**.

## Status

The prototype is operational, including the Streamlit interface, document retrieval, conversational context, validation, and execution metrics. Further work will evaluate adversarial queries, revision behavior, and known limitations such as duplicate source entries and citation-link formatting.

## Data Policy

Only publicly available, synthetic, or anonymized information is used. No confidential or sensitive organizational data is included.
