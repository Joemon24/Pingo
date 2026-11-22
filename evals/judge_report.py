"""Judge saved synthetic evaluation drafts against their generation-time articles."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from evals.failure_artifacts import artifact, write_artifacts
from evals.groundedness import JUDGE_VERSION, JudgeOutputError, judge_draft, make_agents, specificity_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--output", type=Path, default=Path("evals/results/groundedness.json"))
    args = parser.parse_args()
    if not args.live or not args.judge_model.startswith("groq:"):
        parser.error("Explicit --live and groq:<model> required")
    if args.output.exists():
        parser.error("Output already exists")
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    available = {model.id for model in Groq(max_retries=0, timeout=15).models.list().data}
    if args.judge_model.removeprefix("groq:") not in available:
        parser.error("Judge model unavailable to this account")

    raw = args.report.read_bytes()
    report = json.loads(raw)
    extractor, grader = make_agents(args.judge_model)
    results, failures, rejected = [], [], []
    no_generation = 0
    for case in report["results"]:
        evidence = case.get("generation_evidence", {})
        generated = evidence.get("generated_draft")
        if not generated:
            no_generation += 1
            continue
        articles = evidence.get("articles")
        if not articles:
            failures.append({"id": case["id"], "error_type": "MissingArticleSnapshot"})
            continue
        try:
            result = judge_draft(generated["reply"], articles, extractor, grader)
        except Exception as exc:
            failures.append({"id": case["id"], "error_type": type(exc).__name__})
            if isinstance(exc, JudgeOutputError):
                rejected.append(artifact(case["id"], {"draft": generated["reply"], "articles": articles},
                                         exc, "first"))
            continue
        results.append({
            "id": case["id"], **result,
            "specificity_flags": specificity_flags(generated["reply"], articles),
        })
    labels = [row["label"] for result in results for row in result["grade"]["judgments"]]
    unsupported = labels.count("unsupported")
    contradicted = labels.count("contradicted")
    affected = sum(any(row["label"] != "supported" for row in result["grade"]["judgments"]) for result in results)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": sha256(raw).hexdigest(), "dataset_sha256": report["dataset_sha256"],
        "generator_prompt_version": report["prompt_version"],
        "generator_models": report["models"], "judge_model": args.judge_model,
        "judge_version": JUDGE_VERSION,
        "scope": "Model judgments over extracted claims. Human calibration and extraction coverage remain required.",
        "metrics": {
            "judged_drafts": len(results), "extracted_claims": len(labels),
            "unsupported_claims": unsupported, "contradicted_claims": contradicted,
            "unsupported_claim_rate": unsupported / len(labels) if labels else None,
            "unsupported_or_contradicted_claim_rate": (unsupported + contradicted) / len(labels) if labels else None,
            "drafts_with_unsupported_or_contradicted_claims": affected,
            "affected_draft_rate": affected / len(results) if results else None,
            "cases_without_generation": no_generation, "judge_failures": len(failures),
        },
        "failures": failures, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rejected:
        output["validation_artifact_file"] = write_artifacts(
            args.output, rejected, {"judge_model": args.judge_model, "judge_version": JUDGE_VERSION,
                                    "input_sha256": output["input_sha256"]},
        ).name
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["metrics"], indent=2))


if __name__ == "__main__":
    main()
