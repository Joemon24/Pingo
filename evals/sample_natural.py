"""Extract and stratify model-generated claims for blind human labeling."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

from evals.groundedness import make_agents, check_model, validate_inventory


def sample_claims(candidates: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or len(candidates) < n:
        raise ValueError(f"Need {n} claims; only {len(candidates)} available")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for candidate in candidates:
        groups[candidate["category"]].append(candidate)
    for category in sorted(groups):
        rng.shuffle(groups[category])
    selected = []
    # Even allocation across categories, redistributing when a category is exhausted.
    while len(selected) < n:
        for category in sorted(groups):
            if groups[category] and len(selected) < n:
                selected.append(groups[category].pop())
    return selected


def extract_candidates(report: dict, extractor) -> tuple[list[dict], list[dict]]:
    candidates, failures = [], []
    for index, case in enumerate(report["results"]):
        snapshot = case.get("generation_evidence", {})
        generated = snapshot.get("generated_draft")
        if not generated or not generated.get("reply"):
            continue
        articles = snapshot.get("articles")
        if not articles:
            failures.append({"case": case["id"], "error_type": "MissingArticleSnapshot"})
            continue
        try:
            draft = generated["reply"]
            inventory = extractor.run_sync(json.dumps({"draft": draft})).output
            validate_inventory(draft, inventory)
        except Exception as exc:
            failures.append({"case": case["id"], "error_type": type(exc).__name__})
            continue
        for claim in inventory.claims:
            candidates.append({
                "case_id": f"natural-{index+1:03}", "claim_id": claim.id, "text": claim.text,
                "draft": draft, "articles": articles,
                "category": case.get("expected_section") or case.get("section", "unknown"),
            })
    return candidates, failures


def write_sample(selected: list[dict], directory: Path, source_hash: str, seed: int, failures: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    cases = {}
    for row in selected:
        case = cases.setdefault(row["case_id"], {
            "id": row["case_id"], "kind": "natural", "category": row["category"],
            "draft": row["draft"], "articles": row["articles"], "claims": [],
        })
        case["claims"].append({"id": row["claim_id"], "text": row["text"]})
    dataset = {"stratum": "natural", "source_sha256": source_hash, "sampling_seed": seed,
               "sampling_rule": "Even category allocation without replacement; redistribute exhausted strata.",
               "extraction_failures": failures, "cases": list(cases.values())}
    raw = (json.dumps(dataset, indent=2) + "\n").encode()
    (directory / "dataset.json").write_bytes(raw)
    form = {"dataset_sha256": sha256(raw).hexdigest(), "claims": [
        {"id": case["id"] + "/" + claim["id"], "text": claim["text"], "label": None}
        for case in cases.values() for claim in case["claims"]
    ]}
    (directory / "human_labels.template.json").write_text(json.dumps(form, indent=2) + "\n")
    (directory / "human_evidence.json").write_text(json.dumps({"cases": list(cases.values())}, indent=2) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results/natural"))
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("--live is required for extraction")
    if args.output_dir.exists():
        parser.error("Output directory exists; choose a fresh directory")
    if args.n < 1:
        parser.error("--n must be positive")
    raw = args.report.read_bytes()
    report = json.loads(raw)
    check_model(args.judge_model)
    extractor, _ = make_agents(args.judge_model)
    candidates, failures = extract_candidates(report, extractor)
    print("Extraction failures:", len(failures), json.dumps(failures))
    try:
        selected = sample_claims(candidates, args.n, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    write_sample(selected, args.output_dir, sha256(raw).hexdigest(), args.seed, failures)
    print(f"Sampled {len(selected)} claims. Human labels are blank (null).")
    print("Dataset and blind form written to", args.output_dir)


if __name__ == "__main__":
    main()
