"""Replay one saved synthetic troubleshooting email with its generation-time knowledge."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, sleep
from types import SimpleNamespace
from unittest.mock import patch

from app import db
from app.ai import decide as flow
from app.ai.agent import write_reply
from app.ai.grounding import CitationError
from app.ai.prompts import load_prompt
from app.config import settings
from app.gmail import strip_quoted
from app.telemetry import execution
from pydantic_ai.exceptions import ModelHTTPError


def trial(thread, knowledge, version):
    stages = []
    row = {"error": None, "rate_limited": False}

    def generate(thread, section):
        try:
            return write_reply(thread, section, prompt_version=version)
        except CitationError as exc:
            row["error"] = str(exc)
            raise
        except ModelHTTPError as exc:
            row.update(error=type(exc).__name__, rate_limited=exc.status_code == 429)
            raise

    start = perf_counter()
    token = execution.set(stages)
    try:
        classification = SimpleNamespace(section="troubleshooting", needs_human=False)
        with patch.object(flow, "classify", return_value=classification), patch.object(
            db, "get_section", side_effect=knowledge.get
        ), patch.object(flow, "write_reply", generate):
            decision = flow.decide(thread)
    finally:
        execution.reset(token)
    generation = next((s for s in stages if s["stage"] == "generation"), {})
    row.update(
        latency_ms=round(1000 * (perf_counter() - start)), escalated=decision.action == "escalate",
        citation_pass=any(s["stage"] == "citation_check" and s["status"] == "ok"
                          and s.get("citations", 0) > 0 for s in stages),
        input_tokens=generation.get("input_tokens"), output_tokens=generation.get("output_tokens"),
    )
    if row["escalated"] and not row["error"]:
        row["error"] = decision.failure or decision.reason
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--prompt-version", choices=("pingo-v2-citations", "pingo-v3-verbatim-citations"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--interval-seconds", type=int, default=25)
    args = parser.parse_args()
    if args.n < 1 or args.interval_seconds < 1 or settings.delivery_enabled:
        parser.error("Require positive N and pacing, and disabled delivery")
    raw = args.capture.read_bytes()
    capture = json.loads(raw)
    if capture["decision"]["section"] != "troubleshooting":
        parser.error("Expected a troubleshooting capture")
    with db.connect() as conn:
        msg = conn.execute(
            "select t.subject,m.body from events e join messages m using(gmail_message_id) "
            "join tickets t on t.id=m.ticket_id where e.result->>'trace_id'=%s",
            (capture["trace_id"],),
        ).fetchone()
    if not msg:
        parser.error("Captured event has no saved email in this database")
    thread = f"Subject: {msg['subject']}\n\nCustomer:\n{strip_quoted(msg['body'])}"
    knowledge = {}
    for article in capture["generation_evidence"]["articles"]:
        knowledge.setdefault(article["section"], []).append({k:v for k,v in article.items() if k != "section"})
    identity = dict(capture_sha256=sha256(raw).hexdigest(), thread_sha256=sha256(thread.encode()).hexdigest(),
                    version=args.prompt_version, prompt_sha256=sha256(load_prompt(args.prompt_version).encode()).hexdigest(),
                    model=settings.agent_model, n=args.n)
    report = {"scope": "Single saved email, fixed troubleshooting classification and captured articles",
              "identity": identity, "trials": [], "rate_limits": []}
    if args.resume:
        report = json.loads(args.output.read_text())
        if report["identity"] != identity:
            parser.error("Checkpoint inputs differ")
    elif args.output.exists():
        parser.error("Use --resume or a fresh output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = report["trials"]
    while len(rows) < args.n:
        if rows or report["rate_limits"]:
            sleep(args.interval_seconds)
        row = trial(thread, knowledge, args.prompt_version)
        if row["rate_limited"]:
            report["rate_limits"].append(row)
        else:
            rows.append(row)
        report["summary"] = dict(completed=len(rows), requested=args.n,
            citation_pass=sum(r["citation_pass"] for r in rows), escalations=sum(r["escalated"] for r in rows),
            failure_reasons=dict(Counter(r["error"] for r in rows if r["error"])),
            rate_limited=len(report["rate_limits"]), input_tokens=sum(r["input_tokens"] or 0 for r in rows),
            output_tokens=sum(r["output_tokens"] or 0 for r in rows),
            usage_unknown=sum(r["input_tokens"] is None for r in rows),
            latency_ms=[r["latency_ms"] for r in rows])
        tmp = args.output.with_suffix(".partial.tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.output)
        if row["rate_limited"]:
            break
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
