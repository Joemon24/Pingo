
from app.ai.schemas import Issue

CUSTOMER_STEPS = {
    "provide-guidance",
    "ask-details",
}

SAFETY_PROBLEMS = {
    "overheating",
    "swollen-battery",
}


def next_step(issue: Issue) -> str:
    if issue.safety_concern:
        return "escalate-safety"

    if issue.problem in SAFETY_PROBLEMS:
        return "escalate-safety"

    if issue.problem == "damaged-product":
        return "human-inspection"

    if issue.problem == "missing-accessory":
        return "human-fulfillment"

    if issue.product == "unknown":
        return "escalate"

    if issue.problem == "other":
        return "escalate"

    if issue.problem in {
        "not-charging",
        "slow-charging",
        "not-powering-on",
        "pairing-failure",
        "audio-problem",
    }:
        return "provide-guidance"

    return "escalate"
