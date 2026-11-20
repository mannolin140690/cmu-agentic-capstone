# Agentic Analytics Platform Support and Research Assistant

**Final Capstone — Carnegie Mellon University / Emeritus, Agentic AI Program**

## Overview

A multi-agent, retrieval-augmented assistant for answering version-specific technical questions about enterprise analytics platforms. The architecture is intended to extend across products such as **Cloudera, Tableau, Pentaho, and Pyramid Analytics**. The current working prototype indexes **Pentaho Data Integration 9.3** and **Pyramid Analytics** public documentation.

The assistant answers questions in Spanish using English-language documentation, cites retrieved sources, handles conversational follow-ups, and makes its limitations explicit when evidence is insufficient. It is **read-only** and does not access or modify production systems.

## Architecture

The application uses a bounded **LangGraph** workflow:

```text
User → Supervisor → Query Rewriter → Retrieval → Researcher → Validator
            │                                          │          │
            └─ unclear product → Clarification          │          ├─ approved → Answer
                                                       │          └─ rejected → Reviser → Validator
                                                       │                              │
                                                       └──────────────────────────────└─ still rejected → Escalation
```

The **Supervisor** identifies product, version, and question intent. The **Query Rewriter** resolves follow-ups using short-term conversation state. **Retrieval** searches Qdrant semantically for technical questions or scans indexed document metadata for inventory questions. The **Researcher** drafts an evidence-based answer; the **Validator** checks its grounding, citations, and relevance. The system allows at most **one revision**. Persistent validation failures produce a limitation notice and a **simulated human-review handoff**, not an unverified final answer.

**Technology:** Python, LangChain, LangGraph, OpenAI `gpt-4.1-mini`, Pydantic, Qdrant, `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional embeddings), Docker Compose, and Streamlit. Conversation memory uses LangGraph's in-memory checkpointer; request traces capture node timings and token usage. FastAPI, MCP, and Tree-of-Thought troubleshooting are **not part of the implemented prototype**.

## Run Locally

Requirements: Python 3.12, Docker with Compose, and an OpenAI API key. Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then start Qdrant:

```bash
docker compose up -d
```

**Populate the knowledge base (required on first setup):** open `notebooks/01_rag_ingestion_retrieval.ipynb` in Jupyter and run its cells. The notebook downloads and chunks public documentation, embeds it, and loads the `cmu_technical_docs` collection into Qdrant at `localhost:6333`. The first run also downloads the embedding model.

Start the UI:

```bash
PYTHONPATH=. python -m streamlit run app/streamlit_app.py
```

## Usage

Ask questions in Spanish using the Streamlit chat interface. Answers include supporting sources and execution traces. For example:

- ¿Para qué sirve Spoon en Pentaho Data Integration 9.3?
- ¿Qué diferencias hay entre las licencias Professional, Analyst, Viewer y Basic de Pyramid Analytics?
- ¿Qué documentos de Pentaho están indexados en esta base de conocimiento?

Follow-up questions can reuse product and version context within the same session. The assistant will request clarification when it cannot identify a supported product.

## Project Files

| Path | Purpose |
| --- | --- |
| `src/` | LangGraph workflow, agents, Qdrant search, and observability |
| `app/streamlit_app.py` | Interactive chat application |
| `notebooks/01_rag_ingestion_retrieval.ipynb` | Documentation ingestion and indexing |
| `notebooks/02_react_rag_validator.ipynb` | Earlier Researcher–Validator prototype |
| `tests/` | Regression, benchmark, escalation, and architecture comparison scripts |
| `results/` | Saved evaluation metrics and answers |

## Evaluation

The **20-case functional benchmark** completed 20/20 executions, with **100% routing accuracy** and **100% retrieval-strategy accuracy**. All **six regression cases passed**. Mean latency was **7.36 s** (p95 **9.29 s**) and mean consumption was **5,161 tokens/request**. The Validator approved all benchmark responses; **no natural revisions or escalations occurred** in that run.

A **10-question pilot** compared three architectures (30 completed executions):

| Architecture | Mean latency | Mean tokens/question |
| --- | ---: | ---: |
| Standalone LLM | 2.98 s | 293 |
| Simple RAG | 4.12 s | 1,192 |
| Multi-agent RAG | 7.49 s | 5,028 |

Document retrieval improved access to version-specific information compared with the standalone LLM. In this small pilot, the multi-agent approach had higher latency and token costs **without a demonstrated accuracy advantage over simple RAG**. Validator judgments are not independent ground truth; formal answer-quality scoring remains incomplete.

To run the included tests:

```bash
PYTHONPATH=. python tests/test_escalation.py
PYTHONPATH=. python tests/test_regression.py
PYTHONPATH=. python tests/benchmark.py
PYTHONPATH=. python tests/compare_architectures.py
```

The comparison script uses `tests/golden_questions_v1.json`. Benchmark outputs include `results/benchmark_results.json` and `results/benchmark_summary.csv`; the comparison produces `results/architecture_comparison.json`. Running the benchmark and comparisons makes billable OpenAI API requests.

## Reliability and Limitations

The system uses product/version filtering, evidence-grounded prompts, independent LLM-based validation, and a single bounded revision attempt. For persistent validation failures, it suppresses the unverified draft and records a simulated handoff in `logs/handoffs.jsonl` (local runtime artifact, not a live ticketing integration).

Limitations include a small public-documentation corpus, fixed top-3 semantic retrieval without a relevance threshold or reranking, in-memory session state, simulated escalation, and a limited evaluation sample. Further work could add more analytics platforms, hybrid retrieval/reranking, persistent memory, independent quality evaluation, and an operational support-ticket workflow.

## Data Policy

This demonstration uses public documentation and synthetic questions only. No confidential enterprise data or production-system access is required.
