"""Keep rejected synthetic output next to the corresponding local report."""
import json
from pathlib import Path

from evals.groundedness import JudgeOutputError


def artifact(case_id: str, case: dict, error: JudgeOutputError, attempt: str) -> dict:
    return {
        "case_id": case_id,
        "stage": error.stage,
        "failure_type": type(error).__name__,
        "validation_issue": str(error),
        "attempt": attempt,
        "draft": case["draft"],
        "inventory": error.inventory.model_dump()["claims"] if error.inventory else case.get("claims"),
        "articles": error.articles if error.articles is not None else case.get("articles", []),
        "rejected_output": error.rejected_output,
        "cited_article_id": error.evidence.article_id if error.evidence else None,
        "rejected_quote": error.evidence.quote if error.evidence else None,
    }


def write_artifacts(output: Path, records: list[dict], identity: dict) -> Path:
    path = output.with_suffix(".validation-failures.json")
    with path.open("x") as file:
        json.dump({
            "scope": "Rejected output and matching article snapshots from synthetic evaluations.",
            "run": identity,
            "failures": records,
        }, file, indent=2)
    return path
