"""Capture an operator-created synthetic Gmail case and print a blank labeling form."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from evals.failure_artifacts import artifact, write_artifacts
from evals.groundedness import JudgeOutputError, check_model, make_agents, run_stage, validate_inventory
from evals.sample_natural import write_sample


def capture(subject: str) -> dict:
    from app import db
    from app.ai.schemas import PROMPT_VERSION
    from app.config import settings
    from app.gmail import Gmail, get_headers
    from app.pipeline import handle_message
    from app.telemetry import evaluation_capture

    if not settings.dry_run or settings.delivery_enabled:
        raise ValueError("Capture requires DRY_RUN=true, DELIVERY_ENABLED=false,")
    if not re.fullmatch(r"Pingo M0 [A-Za-z0-9 -]{1,80}", subject):
        raise ValueError("Use a unique synthetic subject beginning 'Pingo M0 '")
    gmail = Gmail()
    messages = gmail.users.messages().list(
        userId="me", q=f'in:inbox subject:"{subject}"', maxResults=20,
    ).execute()
    matches = [
        row["id"] for row in messages.get("messages", [])
        if get_headers(gmail.get_message(row["id"])).get("subject") == subject
    ]
    if messages.get("nextPageToken") or len(matches) != 1:
        raise ValueError("Expected one exact inbox match. Use a new unique subject.")
    message_id = matches[0]
    if db.get_event(message_id):
        raise ValueError("This message already has an event. Use a fresh synthetic email.")
    evidence: dict = {}
    token = evaluation_capture.set(evidence)
    try:
        result = handle_message(gmail, message_id)
    finally:
        evaluation_capture.reset(token)
    if not result:
        raise ValueError("This process did not save a decision. Check for another running poller.")
    return {
        "scope": "One operator-created synthetic Gmail message; generation-time evidence.",
        "prompt_version": result["prompt_version"],
        "trace_id": result["trace_id"],
        "decision": {key: result.get(key) for key in ("action", "section", "reply", "sources", "reason", "failure")},
        "stages": result["stages"],
        "generation_evidence": evidence,
    }


def label_form(report_path: Path, output: Path, model: str) -> int:
    raw = report_path.read_bytes()
    report = json.loads(raw)
    if report.get("prompt_version") not in {"pingo-v2-citations", "pingo-v3-verbatim-citations"}:
        raise ValueError("Use a versioned citation-prompt capture")
    if output.exists() or output.with_suffix(".validation-failures.json").exists():
        raise ValueError("Choose a new directory for the blank form and its validation artifact")
    evidence = report["generation_evidence"]
    draft = (evidence.get("generated_draft") or {}).get("reply")
    articles = evidence.get("articles")
    if not draft or not articles:
        raise ValueError("Capture has no generated draft and article snapshot to label")
    check_model(model)
    extractor, _ = make_agents(model)
    case = {"draft": draft, "articles": articles}
    source_hash = sha256(raw).hexdigest()
    payload = {"draft": draft}
    rejected = []
    # This retry is for the interactive form only. Calibration keeps its original prompt and rates.
    try:
        for attempt in ("first", "after_retry"):
            try:
                inventory = run_stage(extractor, json.dumps(payload), "extraction").output
                validate_inventory(draft, inventory)
                break
            except JudgeOutputError as exc:
                rejected.append(artifact("bluetooth-demo", case, exc, attempt))
                payload["validation_feedback"] = (
                    f"The previous extraction failed validation: {exc}. "
                    "Copy each claim as an exact contiguous span from draft, with a unique ID. "
                    "If splitting a sentence would require rewriting it or adding punctuation, "
                    "copy a larger contiguous span and preserve its qualifiers."
                )
        else:
            raise ValueError("Extraction failed validation after one retry")
    finally:
        if rejected:
            output.parent.mkdir(parents=True, exist_ok=True)
            location = write_artifacts(output, rejected, {"source_sha256": source_hash, "judge_model": model})
            print("Extraction validation failures saved to", location)
    if not inventory.claims:
        raise ValueError("Extractor returned zero claims")
    selected = [
        {**case, "case_id": "bluetooth-demo", "claim_id": claim.id, "text": claim.text,
         "category": report["decision"]["section"]}
        for claim in inventory.claims
    ]
    write_sample(selected, output, source_hash, None, [])
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture")
    collect.add_argument("--subject", required=True)
    collect.add_argument("--output", type=Path, required=True)
    form = commands.add_parser("label-form")
    form.add_argument("report", type=Path)
    form.add_argument("--judge-model", required=True)
    form.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "capture":
            if args.output.exists():
                parser.error("Choose a new report filename")
            report = capture(args.subject)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as file:
                json.dump(report, file, indent=2, default=str)
            print(f"Saved {report['prompt_version']} capture to {args.output}")
        else:
            count = label_form(args.report, args.output_dir, args.judge_model)
            print(f"Saved a blank form with {count} claims to {args.output_dir}")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    except Exception as exc:
        # SDK exceptions can include URLs or private configuration.
        parser.error(f"{args.command} failed ({type(exc).__name__})")


if __name__ == "__main__":
    main()
