"""Per-execution evidence, without raw customer or model messages."""
from contextvars import ContextVar
from time import perf_counter
from typing import Any
from uuid import uuid4

evaluation_capture: ContextVar[dict | None] = ContextVar("evaluation_capture", default=None)

execution: ContextVar[list[dict[str, Any]] | None] = ContextVar("execution", default=None)


def record(stage: str, started: float, **details: Any) -> None:
    entries = execution.get()
    if entries is not None:
        entries.append({"stage": stage, "latency_ms": round((perf_counter() - started) * 1000, 2), **details})


def model_run(agent: Any, prompt: str, stage: str) -> Any:
    started = perf_counter()
    try:
        result = agent.run_sync(prompt)
        capture = evaluation_capture.get()
        if capture is not None and stage == "generation":
            capture["generated_draft"] = result.output.model_dump()
        usage = result.usage
        record(stage, started, status="ok", input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
        return result.output
    except Exception as exc:
        record(stage, started, status="failed", error_type=type(exc).__name__)
        raise


def new_trace() -> str:
    return uuid4().hex
