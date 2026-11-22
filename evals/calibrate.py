"""Phase 1B calibration. Human labels are mandatory; no labels are generated."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import sleep

from pydantic_ai.exceptions import ModelQuotaExceededError, ModelRateLimitError

from evals.groundedness import (
    GRADING_PROMPT, EXTRACTION_PROMPT, JUDGE_VERSION, Grade, Inventory, make_agents,
    validate_grade, validate_inventory, specificity_flags, build_grader_input,
    check_model, smoke_agents,
)
from evals.measurement import load_labels, claim_covered, summarize, rate


def retry_rate_limit(call, case_id: str):
    try:
        return call()
    except ModelRateLimitError as exc:
        if exc.retry_after is not None and exc.retry_after > 90:
            raise
        delay = max(1, exc.retry_after) if exc.retry_after is not None else 60
        print(f"{case_id}: rate limited; retrying once in {delay:g}s", file=sys.stderr, flush=True)
        sleep(delay)
        return call()


def case_inventory(case: dict) -> Inventory:
    return Inventory(claims=[{"id": row["id"], "text": row["text"]} for row in case["claims"]])


def calibrate_case(case: dict, grader):
    inventory = case_inventory(case)
    run = grader.run_sync(build_grader_input(case["draft"], inventory, case["articles"]))
    validate_grade(inventory, run.output, case["articles"])
    return run


def reused_results(previous: dict, identity: dict, inputs: list[tuple[dict, dict]]) -> dict:
    for key, value in identity.items():
        if key == "inputs":
            # Copying the exact label file can change mtime without changing its bytes.
            hashes = ("dataset_sha256", "labels_sha256")
            prior = [tuple(row[h] for h in hashes) for row in previous[key]]
            value = [tuple(row[h] for h in hashes) for row in value]
        else:
            prior = previous.get(key)
        if prior != value:
            raise ValueError(f"Resume report has different {key}")

    cases = {
        dataset.get("stratum", "seeded") + "/" + case["id"]: case
        for dataset, _ in inputs for case in dataset["cases"]
    }
    results = {}
    extraction = identity["mode"] == "extraction"
    field = "inventory" if extraction else "grade"
    failed = {entry["case_id"] for entry in previous["failures"]}
    for entry in previous["results"]:
        key = entry["case_id"]
        if key in failed or field not in entry:
            continue
        if key in results or key not in cases:
            raise ValueError(f"Invalid resume case: {key}")
        case = cases[key]
        if extraction:
            validate_inventory(case["draft"], Inventory.model_validate(entry[field]))
        else:
            validate_grade(case_inventory(case), Grade.model_validate(entry[field]), case["articles"])
        results[key] = entry
    return results


def run_calibration(
    inputs: list[tuple[dict, dict]], extractor, grader, extraction: bool,
    *, previous_results: dict | None = None,
) -> dict:
    results, failures, rows = [], [], []
    extraction_counts: dict[str, dict] = {}
    previous_results = previous_results or {}
    rate_limit_blocked = False
    deferred_cases = []
    for dataset, human in inputs:
        stratum = dataset.get("stratum", "seeded")
        if stratum not in {"seeded", "natural"}:
            raise ValueError("Unknown dataset stratum")
        for case in dataset["cases"]:
            key = stratum + "/" + case["id"]
            result = previous_results.get(key, {"case_id": key}).copy()
            predicted = {}
            deferred = rate_limit_blocked and key not in previous_results
            if deferred:
                deferred_cases.append(key)
            if extraction:
                kind = stratum + "/" + case.get("kind", "natural")
                count = extraction_counts.setdefault(kind, {
                    "covered": 0, "claims": 0, "drafts": 0, "failed_drafts": 0, "deferred_drafts": 0,
                })
                count["drafts"] += 1
                count["claims"] += len(case["claims"])
                count["deferred_drafts"] += int(deferred)
            if not deferred:
                try:
                    if extraction:
                        if key in previous_results:
                            inventory = Inventory.model_validate(result["inventory"])
                        else:
                            payload = json.dumps({"draft": case["draft"]})
                            inventory = retry_rate_limit(lambda: extractor.run_sync(payload), key).output
                            validate_inventory(case["draft"], inventory)
                            result["inventory"] = inventory.model_dump()
                        count["covered"] += sum(
                            any(claim_covered(claim["text"], found.text) for found in inventory.claims)
                            for claim in case["claims"]
                        )
                    else:
                        if key in previous_results:
                            grade = Grade.model_validate(result["grade"])
                        else:
                            run = retry_rate_limit(lambda: calibrate_case(case, grader), key)
                            grade = run.output
                            result["grade"] = grade.model_dump()
                            result["usage"] = {
                                "input_tokens": run.usage.input_tokens, "output_tokens": run.usage.output_tokens,
                            }
                        predicted = {row.claim_id: row.label for row in grade.judgments}
                except Exception as exc:
                    if extraction:
                        count["failed_drafts"] += 1
                    failures.append({
                        "case_id": key, "stage": "extraction" if extraction else "grading",
                        "error_type": type(exc).__name__,
                    })
                    if isinstance(exc, (ModelRateLimitError, ModelQuotaExceededError)):
                        rate_limit_blocked = True
                        print(f"{key}: {type(exc).__name__}; deferring remaining model calls",
                              file=sys.stderr, flush=True)
            for claim in case["claims"]:
                flags = specificity_flags(claim["text"], case["articles"])
                rows.append({
                    "id": key + "/" + claim["id"], "stratum": stratum,
                    "expected": human[case["id"] + "/" + claim["id"]],
                    "actual": predicted.get(claim["id"]), "flagged": bool(flags), "flags": flags,
                })
            results.append(result)
    for count in extraction_counts.values():
        count["recall"] = rate(count["covered"], count["claims"])
    return {
        "failures": failures, "failure_types": dict(Counter(row["stage"] + "/" + row["error_type"] for row in failures)),
        "judge_failure_count": sum(row["stage"] == "grading" for row in failures),
        "reused_case_count": len(previous_results),
        "deferred_cases": deferred_cases,
        "metrics": summarize(rows) if not extraction else None,
        "extraction": extraction_counts if extraction else None,
        "extraction_rule": "Casefolded word-token span containment, or >=90% multiset seed-token recall against one extracted claim. Extra words allowed; this measures span coverage, not atomicity or entailment.",
        "claim_results": rows, "results": results,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--human-labels", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=Path("evals/results/judge-calibration.json"))
    parser.add_argument("--resume", type=Path, help="Reuse successful cases from a matching calibration report")
    parser.add_argument("--extraction", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Only one tiny live request per agent; no calibration")
    args = parser.parse_args(argv)
    if not args.live or not args.judge_model.startswith("groq:"):
        parser.error("Use --live and an explicit groq:<model ID>")
    if args.smoke:
        if args.extraction or args.dataset or args.human_labels or args.resume:
            parser.error("--smoke is a standalone diagnostic")
        check_model(args.judge_model)
        if not smoke_agents(*make_agents(args.judge_model)):
            raise SystemExit(1)
        return
    datasets = args.dataset or [Path("evals/groundedness_seeds.json")]
    if not args.human_labels or len(args.human_labels) != len(datasets):
        parser.error("Live calibration requires one --human-labels file per dataset")
    inputs, metadata = [], []
    try:
        for dataset_path, label_path in zip(datasets, args.human_labels):
            dataset, human, meta = load_labels(dataset_path, label_path)
            inputs.append((dataset, human))
            metadata.append(meta)
        if len({dataset.get("stratum", "seeded") for dataset, _ in inputs}) != len(inputs):
            raise ValueError("Provide at most one dataset per stratum")
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error("Output exists. Choose a new --output to preserve previous evidence.")
    identity = dict(
        judge_model=args.judge_model,
        judge_version=JUDGE_VERSION, inputs=metadata,
        prompt_sha256=sha256(GRADING_PROMPT.encode()).hexdigest(),
        extraction_prompt_sha256=sha256(EXTRACTION_PROMPT.encode()).hexdigest(),
        mode="extraction" if args.extraction else "fixed_inventory_grading",
    )
    previous_results = {}
    resume_hash = None
    if args.resume:
        try:
            raw = args.resume.read_bytes()
            previous_results = reused_results(json.loads(raw), identity, inputs)
            resume_hash = sha256(raw).hexdigest()
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
    check_model(args.judge_model)
    extractor, grader = make_agents(args.judge_model)
    report = run_calibration(inputs, extractor, grader, args.extraction, previous_results=previous_results)
    report.update(
        **identity, created_at=datetime.now(timezone.utc).isoformat(),
        resumed_from_sha256=resume_hash,
        interval_caveat="Wilson intervals treat claims as independent. Claims within drafts are correlated; intervals are descriptive.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("Judge failures:", report["judge_failure_count"])
    print("Failure types:", json.dumps(report["failure_types"]))
    print("Cases reused:", report["reused_case_count"])
    print("Cases deferred:", len(report["deferred_cases"]))
    if args.extraction:
        print("Extraction failures:", len(report["failures"]))
    print(json.dumps(report["extraction"] if args.extraction else report["metrics"], indent=2))


if __name__ == "__main__":
    main()
