
import json
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler


LOG_PATH = Path("logs/agent_traces.jsonl")


class AgentObservability(BaseCallbackHandler):

    def __init__(self):
        super().__init__()

        self.trace_id = str(uuid.uuid4())
        self.started_at = time.perf_counter()

        self.node_starts = {}
        self.node_durations = {}

        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

        self._lock = Lock()

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs
    ):
        metadata = metadata or {}
        tags = tags or []

        node_name = metadata.get("langgraph_node")

        if not node_name:
            return

        # Ignore nested LangChain executions.
        # Only register actual LangGraph node runs.
        is_graph_step = any(
            str(tag).startswith("graph:step:")
            for tag in tags
        )

        if not is_graph_step:
            return

        with self._lock:
            self.node_starts[run_id] = (
                node_name,
                time.perf_counter()
            )

    def on_chain_end(
        self,
        outputs,
        *,
        run_id,
        **kwargs
    ):
        with self._lock:
            started = self.node_starts.pop(
                run_id,
                None
            )

            if started is None:
                return

            node_name, start_time = started

            duration = (
                time.perf_counter() - start_time
            )

            self.node_durations.setdefault(
                node_name, []
            ).append(
                round(duration, 4)
            )

    def on_chain_error(
        self,
        error,
        *,
        run_id,
        **kwargs
    ):
        # Clean up runs that failed.
        with self._lock:
            started = self.node_starts.pop(
                run_id,
                None
            )

            if started is None:
                return

            node_name, start_time = started

            duration = (
                time.perf_counter() - start_time
            )

            self.node_durations.setdefault(
                node_name, []
            ).append(
                round(duration, 4)
            )

    def on_llm_end(
        self,
        response,
        *,
        run_id,
        **kwargs
    ):
        usage = (
            response.llm_output or {}
        ).get("token_usage", {})

        if not usage and response.generations:
            message = response.generations[0][0].message

            usage = getattr(
                message,
                "usage_metadata",
                {}
            ) or {}

        with self._lock:
            self.llm_calls += 1

            self.input_tokens += (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or 0
            )

            self.output_tokens += (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or 0
            )

    def summary(self, result, question):

        elapsed = (
            time.perf_counter() - self.started_at
        )

        metrics = result.get("metrics") or {}

        validation = (
            result.get("final_validation")
            or result.get("validation")
            or {}
        )

        if hasattr(validation, "model_dump"):
            validation = validation.model_dump()

        with self._lock:
            node_durations = {
                name: list(durations)
                for name, durations
                in self.node_durations.items()
            }

            llm_calls = self.llm_calls
            input_tokens = self.input_tokens
            output_tokens = self.output_tokens

        return {
            "trace_id": self.trace_id,

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "question": question,
            "product": result.get("product"),
            "intent": result.get("intent"),

            "latency_seconds": round(
                elapsed, 4
            ),

            "node_durations": node_durations,

            "llm_calls": llm_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "total_tokens": (
                input_tokens + output_tokens
            ),

            "retrieval_strategy": metrics.get(
                "retrieval_strategy"
            ),

            "inventory_complete": metrics.get(
                "inventory_complete"
            ),

            "documents_retrieved": len(
                result.get("evidence") or []
            ),

            "revision_count": result.get(
                "revision_count", 0
            ),

            "escalated": bool(
                result.get("escalated")
            ),

            "validation": validation
        }

    def save(self, record):

        LOG_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with LOG_PATH.open(
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str
                ) + "\n"
            )
