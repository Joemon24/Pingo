import logging
import re

import groq
import psycopg
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from app.ai.grounding import CitationError
from app.ai.agent import UNSUPPORTED_CLAIM, write_reply
from app.ai.classifier import classify
from app.ai.rules import CUSTOMER_STEPS, next_step
from app.ai.schemas import Decision, Section

logger = logging.getLogger(__name__)

REVIEW_RULES: tuple[tuple[str, Section, str], ...] = (
    (r"overheat\w*|\b(?:swoll\w*|swell\w*|bulg\w*|smok\w*|sparks?|fire|shock(?:ed|s)?|burn\w*|injur\w*|explod\w*|leak\w*|too hot|very hot|extremely hot)\b",
     "troubleshooting", "Product safety or injury"),
    (r"\b(?:refund\w*|charge\s?back\w*|money back|payment dispute|payment reversal|compensation)\b",
     "returns-refunds", "Financial review"),
    (r"\b(?:lawyer|legal|lawsuit|sue|regulat\w*|consumer rights)\b",
     "other", "Legal review"),
    (r"\b(?:passwords?|otp|one[- ]time (?:password|code)|api keys?|credentials|secrets?|"
     r"credit card|card number|bank account|passport|identity document|personal data)\b",
     "other", "Sensitive information"),
    (r"\b(?:ignore|override|bypass|disregard|forget)\b.{0,80}\b(?:instructions|rules|policy|review)\b"
     r"|\b(?:system prompt|developer message|reveal your prompt)\b",
     "other", "Suspicious instructions"),
    (r"\b(?:cancel (?:my|the) order|change (?:my|the) address|where is my (?:order|package)|"
     r"my tracking (?:number|link))\b",
     "shipping", "Order-specific request"),
)


def decide(
    thread: str, has_attachments: bool = False, *, latest_message: str | None = None
) -> Decision:
    """Return a review decision or a draft. The caller owns persistence."""
    section: Section = "other"

    def escalate(reason: str, **details) -> Decision:
        return Decision(action="escalate", section=section, reply="", reason=reason, **details)

    normalized = " ".join((latest_message if latest_message is not None else thread).casefold().split())
    for pattern, category, reason in REVIEW_RULES:
        if re.search(pattern, normalized):
            section = category
            return escalate(reason)

    if has_attachments:
        return escalate("Attachment requires review")

    try:
        classification = classify(thread)
        section = classification.section
        if classification.needs_human or section == "other":
            return escalate("Classification requires review")

        answer = write_reply(thread, section)
    except CitationError:
        logger.warning("Draft citation integrity check failed")
        return escalate("Draft citations require human review", failure="citation")
    except UnexpectedModelBehavior:
        logger.warning("Model returned invalid structured output")
        return escalate("Invalid structured output", failure="structured-output")
    except ModelHTTPError as exc:
        logger.warning("Model request failed with HTTP %s", exc.status_code)
        return escalate("Model service unavailable", failure="provider")
    except groq.APIError:
        logger.warning("Groq connection or service failure")
        return escalate("Model service unavailable", failure="provider")
    except psycopg.Error:
        logger.warning("Knowledge lookup failed")
        return escalate("Knowledge lookup unavailable", failure="database")

    if answer is None:
        return escalate("Required knowledge section is missing", failure="missing-knowledge")
    if answer.action == "escalate" or not answer.enough_information:
        return escalate("Draft requires human review")
    if not answer.reply.strip() or answer.promises_an_action or UNSUPPORTED_CLAIM.search(answer.reply):
        return escalate("Draft is empty or contains an unsupported commitment")
    if answer.issue:
        step = next_step(answer.issue)
        if step not in CUSTOMER_STEPS:
            return escalate("Product issue requires human review", proposed_action=step)

    return Decision(action="reply", section=section, reply=answer.reply, reason="Draft ready for review", sources=answer.sources)
