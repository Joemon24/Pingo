"""Run with python -m evals.run --live. Writes only synthetic evaluation data."""
import argparse
import json
from pathlib import Path
from time import perf_counter
from hashlib import sha256
from datetime import datetime, timezone


def classification_metrics(expected: list[str], actual: list[str]) -> dict:
    labels = sorted(set(expected) | set(actual))
    rows = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(expected, actual))
        fp = sum(a != label and b == label for a, b in zip(expected, actual))
        fn = sum(a == label and b != label for a, b in zip(expected, actual))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        rows[label] = {
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "support": expected.count(label),
        }
    return {
        "accuracy": sum(a == b for a, b in zip(expected, actual)) / len(expected) if expected else None,
        "per_class": rows,
        "macro_f1": sum(row["f1"] for row in rows.values()) / len(rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("evals/results/latest.json"))
    from app.ai.prompts import VERSIONS
    parser.add_argument("--prompt-version", choices=VERSIONS, default="pingo-v2-citations")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new --output")
    if not args.live:
        parser.error("--live is required to call Groq using synthetic cases")

    from app.ai.decide import decide
    from app.ai import decide as flow
    from app.ai.agent import write_reply
    from functools import partial
    from app.ai.prompts import load_prompt
    from app.config import settings
    from app.telemetry import execution, evaluation_capture
    from app import db

    root = Path(__file__).resolve().parents[1]
    dataset = (root / "evals/golden.json").read_bytes()
    cases = json.loads(dataset)
    # Evaluation process only: explicit prompt selection without changing production settings.
    flow.write_reply = partial(write_reply, prompt_version=args.prompt_version)
    # Require real PostgreSQL retrieval for this baseline.
    db.get_section("products")
    results = []
    for case in cases:
        stages = []
        capture = {}
        capture_token = evaluation_capture.set(capture)
        token = execution.set(stages)
        started = perf_counter()
        try:
            decision = decide(case["message"])
        finally:
            execution.reset(token)
            evaluation_capture.reset(capture_token)
        text = decision.reply.casefold()
        results.append({
            "id": case["id"], "expected_section": case["section"],
            "section": decision.section, "expected_action": case["action"],
            "action": decision.action, "failure": decision.failure,
            "reply": decision.reply, "sources": [source.model_dump() for source in decision.sources], "safety": bool(case.get("safety")),
            "keyword_checks_pass": (
                all(word in text for word in case.get("required", []))
                and not any(word in text for word in case.get("unsupported", []))
            ) if decision.action == "reply" else None,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "stages": stages, "generation_evidence": capture,
        })
    metrics = classification_metrics(
        [row["expected_section"] for row in results], [row["section"] for row in results],
    )
    safety = [row for row in results if row["safety"]]
    model_stages = [stage for row in results for stage in row["stages"]
                    if stage["stage"] in {"classification", "generation"}]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256(dataset).hexdigest(),
        "models": {"classifier": settings.classifier_model, "reply": settings.agent_model},
        "prompt_version": args.prompt_version,
        "prompt_sha256": sha256(
            load_prompt(args.prompt_version).encode()
        ).hexdigest(),
        "knowledge_sha256": sha256(json.dumps([
            {"section": section, "entries": db.get_section(section)}
            for section in ["products", "troubleshooting", "brand-and-tone", "escalation-rules",
                            "returns-refunds", "shipping", "warranty"]
        ], sort_keys=True).encode()).hexdigest(),
        "human_grading": {"correctness": None, "groundedness": None, "retrieval_relevance": None},
        "scope": "Full routing, including deterministic rules; keyword checks require human grading for factual accuracy.",
        "metrics": {
            "routing": metrics,
            "safety_recall": sum(row["action"] == "escalate" for row in safety) / len(safety) if safety else None,
            "structured_stage_success": sum(stage["status"] == "ok" for stage in model_stages) / len(model_stages) if model_stages else None,
            "task_routing_success": sum(row["action"] == row["expected_action"] and not row["failure"] for row in results) / len(results),
        },
        "results": results,
    }
    from psycopg.types.json import Jsonb

    with db.connect() as conn:
        conn.execute("insert into evaluation_runs (report) values (%s)", (Jsonb(report),))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
