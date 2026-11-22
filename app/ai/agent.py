import json
from hashlib import sha256
import re
from time import perf_counter

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.groq import GroqModelSettings

from app import db
from app.ai.grounding import verify_sources
from app.ai.schemas import AgentReply, Section, PROMPT_VERSION
from app.ai.prompts import load_prompt
from app.config import settings
from app.telemetry import evaluation_capture, model_run, record

support_agent = Agent(
    settings.agent_model,
    output_type=NativeOutput(AgentReply, strict=False),
    instructions=load_prompt(PROMPT_VERSION),
    model_settings=GroqModelSettings(timeout=25, max_retries=0),
    retries={"output": 1},
)

UNSUPPORTED_CLAIM = re.compile(
    r"\b(?:will|shall|going to|guarantee to|already|have|has|we(?:’|')ll)\b"
    r"[^.!?\n]{0,60}\b(?:refund(?:ed)?|replac(?:e|ed|ement)|ship(?:ped)?|dispatch(?:ed)?)\b"
    r"|\b(?:refund|replacement|shipment)\b[^.!?\n]{0,40}"
    r"\b(?:approved|issued|processed|confirmed|on its way)\b",
    re.IGNORECASE,
)


def write_reply(thread: str, section: Section, *, prompt_version: str = PROMPT_VERSION) -> AgentReply | None:
    sections = ("brand-and-tone", "escalation-rules", section)
    if section == "troubleshooting":
        sections += ("products",)
    started = perf_counter()
    knowledge = {}
    for name in sections:
        entries = db.get_section(name)
        if not entries or not any(entry["content"].strip() for entry in entries):
            record("retrieval", started, status="missing", section=name)
            return None
        knowledge[name] = entries

    record("retrieval", started, status="ok", sections=list(knowledge),
           sources=[{"section": name, "title": entry["title"], "sha256": sha256(entry["content"].encode()).hexdigest()} for name, entries in knowledge.items() for entry in entries])
    capture = evaluation_capture.get()
    if capture is not None:
        capture["articles"] = [
            {"section": name, **entry} for name, entries in knowledge.items() for entry in entries
        ]
    prompt = json.dumps({"company_knowledge": knowledge, "untrusted_customer_thread": thread})
    agent = support_agent
    if prompt_version != PROMPT_VERSION:
        agent = Agent(settings.agent_model, output_type=NativeOutput(AgentReply, strict=False),
                      instructions=load_prompt(prompt_version),
                      model_settings=GroqModelSettings(timeout=25, max_retries=0), retries={"output": 1})
    answer = model_run(agent, prompt, "generation")
    if prompt_version == "pingo-v1-reconstructed":
        return answer
    started = perf_counter()
    try:
        verify_sources(answer, [entry for entries in knowledge.values() for entry in entries])
    except ValueError:
        record("citation_check", started, status="failed")
        raise
    record("citation_check", started, status="ok", citations=len(answer.sources))
    return answer
