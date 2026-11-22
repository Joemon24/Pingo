import logging
from email.utils import parseaddr

from app import db
from app.ai.decide import decide
from app.ai.schemas import PROMPT_VERSION
from app.config import settings
from app.gmail import (
    ESCALATED_LABEL,
    PROCESSED_LABEL,
    Gmail,
    format_thread,
    get_body,
    get_headers,
    has_attachments,
)
from app.guards import skip_reason
from app.telemetry import execution, new_trace

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 3


def require_dry_run() -> None:
    if not settings.dry_run:
        raise RuntimeError("Pingo requires DRY_RUN=true")


def handle_message(gmail: Gmail, message_id: str) -> dict | None:
    """Persist a draft or review decision, then reconcile Gmail labels."""
    require_dry_run()
    event = db.start_event(message_id, MAX_ATTEMPTS)
    if event is None:
        existing = db.get_event(message_id)
        if existing and existing["status"] in {"processed", "skipped"}:
            apply_labels(gmail, message_id, existing["result"] or {})
        elif existing and existing["status"] == "failed" and existing["attempts"] >= MAX_ATTEMPTS:
            logger.warning("Event exhausted processing attempts and requires review")
            apply_labels(gmail, message_id, {"action": "escalate"})
        return

    entries: list[dict] = []
    token = execution.set(entries)
    trace_id = new_trace()
    try:
        status, result = process(gmail, message_id)
        result.update(trace_id=trace_id, stages=entries, prompt_version=PROMPT_VERSION)
        saved = db.finish_event(event["id"], event["attempts"], status, result)
    except Exception as exc:
        logger.error("Processing attempt %s failed (%s)", event["attempts"], type(exc).__name__)
        db.fail_event(event["id"], event["attempts"], "Processing failed")
        raise
    finally:
        execution.reset(token)

    if not saved:
        logger.info("Discarded result from superseded processing attempt")
        return

    apply_labels(gmail, message_id, result)
    logger.info("Dry-run processing completed with status=%s", status)
    return result


def apply_labels(gmail: Gmail, message_id: str, result: dict) -> None:
    if result.get("action") == "escalate":
        gmail.add_label(message_id, ESCALATED_LABEL)
    gmail.add_label(message_id, PROCESSED_LABEL)


def process(gmail: Gmail, message_id: str) -> tuple[str, dict]:
    require_dry_run()
    message = gmail.get_message(message_id)
    headers = get_headers(message)
    reason = skip_reason(message, headers, gmail.address)
    if reason:
        return "skipped", {"reason": reason}

    sender = parseaddr(headers.get("from", ""))[1]
    ticket_id = db.upsert_ticket(message["threadId"], sender, headers.get("subject", ""))
    db.save_message(ticket_id, message_id, "in", sender, get_body(message))
    thread = gmail.get_thread(message["threadId"])

    if db.ticket_needs_review(ticket_id) or gmail.is_escalated(thread):
        return "skipped", {"reason": "Thread already escalated", "action": "escalate"}

    latest = f"{headers.get('subject', '')}\n{get_body(message)}"
    decision = decide(
        format_thread(thread, gmail.address),
        has_attachments=has_attachments(message),
        latest_message=latest,
    )
    return "processed", {
        **decision.model_dump(),
        "ticket_id": ticket_id,
        "dry_run": True,
        "send_enabled": False,
    }
