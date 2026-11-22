"""Diagnose a rejected seeded quote without changing validation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from time import sleep
from unicodedata import normalize

from pydantic_ai.exceptions import ModelQuotaExceededError, ModelRateLimitError

from evals.calibrate import calibrate_case, case_inventory, retry_rate_limit
from evals.failure_artifacts import artifact, write_artifacts
from evals.groundedness import JudgeOutputError, check_model, make_agents, validate_inventory
from evals.measurement import rate


def formatting_tokens(text: str) -> list[str]:
    # Diagnostic only. Keep words, signs, and decimal/number separators to rule out changed facts.
    return re.findall(r"[+−-]?[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+", normalize("NFKC", text))


def has_formatting_match(quote: str, content: str) -> bool:
    tokens = formatting_tokens(quote)
    full = formatting_tokens(content)
    return bool(tokens) and any(
        full[index:index + len(tokens)] == tokens for index in range(len(full) - len(tokens) + 1)
    )


def classify_artifact(record: dict, seed: dict) -> str:
    articles = seed["articles"]
    canonical = {article["id"]: article for article in articles}
    try:
        validate_inventory(seed["draft"], case_inventory(seed))
        if len(canonical) != len(articles) or any(not article["content"].strip() for article in articles):
            return "seed-data bug"
    except ValueError:
        return "seed-data bug"

    captured = {article["id"]: article for article in record["articles"]}
    if (captured != canonical or len(captured) != len(record["articles"])
            or record["draft"] != seed["draft"]):
        return "snapshot/ID mismatch"
    if record["stage"] == "grading" and record["inventory"] != case_inventory(seed).model_dump()["claims"]:
        return "snapshot/ID mismatch"

    quote, cited = record["rejected_quote"], record["cited_article_id"]
    if not quote or cited is None:
        raise ValueError("This failure has no rejected citation to classify")
    if cited in captured and quote in captured[cited]["content"]:
        raise ValueError("The cited quote is present in the captured snapshot")
    if any(quote in article["content"] for article_id, article in captured.items() if article_id != cited):
        return "correct quote, wrong article ID"
    if cited in captured and has_formatting_match(quote, captured[cited]["content"]):
        return "near-verbatim quote"
    return "fabricated quote"


def repeat_case(case: dict, grader, repeats: int, interval_seconds: float) -> tuple[dict, list[dict]]:
    attempts = {"first": 0, "after_retry": 0}
    failed = {"first": 0, "after_retry": 0}
    rejected, failures = [], []
    successes = 0
    key = "seeded/" + case["id"]
    for index in range(repeats):
        if index and interval_seconds:
            sleep(interval_seconds)
        phases: list[str] = []
        try:
            retry_rate_limit(lambda: calibrate_case(case, grader), key, interval_seconds, phases)
            successes += 1
        except Exception as exc:
            failures.append({"repeat": index + 1, "error_type": type(exc).__name__})
            if isinstance(exc, JudgeOutputError):
                phase = phases[-1]
                failed[phase] += 1
                rejected.append({"repeat": index + 1, **artifact(key, case, exc, phase)})
            if isinstance(exc, (ModelRateLimitError, ModelQuotaExceededError)):
                break
        finally:
            for phase in phases:
                attempts[phase] += 1
    attempted = successes + len(failures)
    return {
        "scope": "Repeated fixed-inventory grading of one synthetic case under strict validation.",
        "case_id": key, "requested": repeats, "attempted": attempted, "deferred": repeats - attempted,
        "valid_results": successes, "failures": failures,
        "validation_failures": {"strict": len(rejected), "normalized": None},
        "validation_rates": {phase: rate(failed[phase], attempts[phase]) for phase in attempts},
    }, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_file", nargs="?", type=Path)
    parser.add_argument("--case-id", default="case-07")
    parser.add_argument("--dataset", type=Path, default=Path("evals/groundedness_seeds.json"))
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--interval-seconds", type=float, default=25)
    parser.add_argument("--output", type=Path, default=Path("evals/results/case-07-repeat.json"))
    args = parser.parse_args()
    dataset_raw = args.dataset.read_bytes()
    if args.repeat < 1 or not 0 <= args.interval_seconds <= 90:
        parser.error("--repeat must be positive and --interval-seconds must be between 0 and 90")
    cases = {case["id"]: case for case in json.loads(dataset_raw)["cases"]}
    case_id = args.case_id.removeprefix("seeded/")
    if case_id not in cases:
        parser.error("Case not present in the dataset")
    case = cases[case_id]
    fingerprint = sha256(dataset_raw).hexdigest()
    if args.artifact_file:
        data = json.loads(args.artifact_file.read_text())
        known_hashes = {row["dataset_sha256"] for row in data["run"].get("inputs", [])}
        if known_hashes and fingerprint not in known_hashes:
            parser.error("Supply the exact seed dataset used by the artifact")
        records = [record for record in data["failures"] if record["case_id"] == "seeded/" + case_id]
        if not records:
            parser.error("No artifact for this seeded case")
        for record in records:
            try:
                category = classify_artifact(record, case)
            except ValueError as exc:
                parser.error(str(exc))
            print(json.dumps({"scope": "Diagnostic only. Strict validation remains active.",
                              "case_id": record["case_id"], "classification": category}))
    if args.live:
        if not args.judge_model or not args.judge_model.startswith("groq:"):
            parser.error("--live requires --judge-model groq:<model ID>")
        if args.output.exists():
            parser.error("Output exists. Choose a fresh path.")
        check_model(args.judge_model)
        _, grader = make_agents(args.judge_model)
        report, artifacts = repeat_case(case, grader, args.repeat, args.interval_seconds)
        report.update(judge_model=args.judge_model, dataset_sha256=fingerprint)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if artifacts:
            identity = {"judge_model": args.judge_model, "inputs": [{"dataset_sha256": fingerprint}]}
            report["validation_artifact_file"] = write_artifacts(args.output, artifacts, identity).name
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    elif not args.artifact_file:
        parser.error("Supply an artifact file or use --live to collect one")


if __name__ == "__main__":
    main()
