
from typing import Literal
from pydantic import BaseModel, Field

PROMPT_VERSION = "pingo-v2-citations"

Section = Literal[
    "products",
    "troubleshooting",
    "returns-refunds",
    "shipping",
    "warranty",
    "other",
]

Product = Literal[
    "usb-c-charger",
    "power-bank",
    "wireless-earbuds",
    "unknown",
]

Problem = Literal[
    "not-charging",
    "slow-charging",
    "not-powering-on",
    "pairing-failure",
    "audio-problem",
    "damaged-product",
    "overheating",
    "swollen-battery",
    "missing-accessory",
    "other",
]


class Classification(BaseModel):
    section: Section
    language: str
    needs_human: bool
    reason: str
    acknowledgement: str


class Issue(BaseModel):
    product: Product
    problem: Problem
    order_number: str | None = None
    model_number: str | None = None
    troubleshooting_attempted: bool = False
    safety_concern: bool = False


class ClaimSource(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)
    article_id: int = Field(gt=0)
    snippet: str = Field(min_length=1, max_length=4000)


class AgentReply(BaseModel):
    action: Literal["reply", "escalate"]
    reply: str
    reasoning: str
    enough_information: bool
    promises_an_action: bool
    issue: Issue | None = None
    sources: list[ClaimSource] = Field(default_factory=list, max_length=40)


class Decision(BaseModel):
    action: Literal["reply", "escalate"]
    section: Section
    reply: str
    reason: str
    proposed_action: str | None = None
    failure: Literal["provider", "structured-output", "database", "missing-knowledge", "citation"] | None = None
    sources: list[ClaimSource] = Field(default_factory=list, max_length=40)
