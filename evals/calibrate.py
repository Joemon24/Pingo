"""Phase 1B calibration. Human labels are mandatory; no labels are generated."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from evals.groundedness import (
    GRADING_PROMPT, EXTRACTION_PROMPT, JUDGE_VERSION, Inventory, make_agents,
    validate_grade, validate_inventory, specificity_flags, build_grader_input,
    check_model, smoke_agents,
)
from evals.measurement import load_labels, claim_covered, summarize, rate


def calibrate_case(case: dict, grader):
    inventory = Inventory(claims=[{"id": row["id"], "text": row["text"]} for row in case["claims"]])
    run = grader.run_sync(build_grader_input(case["draft"], inventory, case["articles"]))
    validate_grade(inventory, run.output, case["articles"])
    return run


def run_calibration(inputs: list[tuple[dict, dict]], extractor, grader, extraction: bool) -> dict:
    results, failures, rows = [], [], []
    extraction_counts: dict[str, dict] = {}
    for dataset, human in inputs:
        stratum = dataset.get("stratum", "seeded")
        if stratum not in {"seeded", "natural"}:
            raise ValueError("Unknown dataset stratum")
        for case in dataset["cases"]:
            key = stratum + "/" + case["id"]
            result = {"case_id": key}
            predicted = {}
            if extraction:
                kind = stratum + "/" + case.get("kind", "natural")
                count = extraction_counts.setdefault(kind, {"covered": 0, "claims": 0, "drafts": 0, "failed_drafts": 0})
                count["drafts"] += 1
                count["claims"] += len(case["claims"])
                try:
                    inventory = extractor.run_sync(json.dumps({"draft": case["draft"]})).output
                    validate_inventory(case["draft"], inventory)
                    count["covered"] += sum(
                        any(claim_covered(claim["text"], found.text) for found in inventory.claims)
                        for claim in case["claims"]
                    )
                    result["inventory"] = inventory.model_dump()
                except Exception as exc:
                    count["failed_drafts"] += 1
                    failures.append({"case_id": key, "stage": "extraction", "error_type": type(exc).__name__})
            else:
                try:
                    run = calibrate_case(case, grader)
                    predicted = {row.claim_id: row.label for row in run.output.judgments}
                    result["grade"] = run.output.model_dump()
                    result["usage"] = {"input_tokens": run.usage.input_tokens, "output_tokens": run.usage.output_tokens}
                except Exception as exc:
                    failures.append({"case_id": key, "stage": "grading", "error_type": type(exc).__name__})
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
    parser.add_argument("--extraction", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Only one tiny live request per agent; no calibration")
    args = parser.parse_args(argv)
    if not args.live or not args.judge_model.startswith("groq:"):
        parser.error("Use --live and an explicit groq:<model ID>")
    if args.smoke:
        if args.extraction or args.dataset or args.human_labels:
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
    check_model(args.judge_model)
    extractor, grader = make_agents(args.judge_model)
    report = run_calibration(inputs, extractor, grader, args.extraction)
    report.update(
        created_at=datetime.now(timezone.utc).isoformat(), judge_model=args.judge_model,
        judge_version=JUDGE_VERSION, inputs=metadata,
        prompt_sha256=sha256(GRADING_PROMPT.encode()).hexdigest(),
        extraction_prompt_sha256=sha256(EXTRACTION_PROMPT.encode()).hexdigest(),
        mode="extraction" if args.extraction else "fixed_inventory_grading",
        interval_caveat="Wilson intervals treat claims as independent. Claims within drafts are correlated; intervals are descriptive.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("Judge failures:", report["judge_failure_count"])
    print("Failure types:", json.dumps(report["failure_types"]))
    print(json.dumps(report["extraction"] if args.extraction else report["metrics"], indent=2))


if __name__ == "__main__":
    main()
