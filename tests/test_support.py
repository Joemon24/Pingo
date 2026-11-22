import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import groq
import httpx
import psycopg
import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

CASES = json.loads((Path(__file__).parents[1] / "evals/golden.json").read_text())


@pytest.fixture
def flow():
    return importlib.import_module("app.ai.decide")


def answer(**changes):
    from app.ai.schemas import AgentReply

    return AgentReply.model_validate({
        "action": "reply", "reply": "Check Bluetooth. Gears and Co Support",
        "reasoning": "Known steps", "enough_information": True,
        "promises_an_action": False,
        "sources": [{"claim": "Check Bluetooth.", "article_id": 1, "snippet": "KB article"}],
        **changes,
    })


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_golden_routing(case, flow, monkeypatch):
    from app.ai.schemas import Classification

    classify = Mock(return_value=Classification(
        section=case["section"], language="en",
        needs_human=case["id"] == "bluetooth-basic-steps-exhausted",
        reason="fixture", acknowledgement="fixture",
    ))
    write = Mock(return_value=answer(reply=" ".join(case.get("required", ["Draft"]))))
    monkeypatch.setattr(flow, "classify", classify)
    monkeypatch.setattr(flow, "write_reply", write)
    decision = flow.decide(case["message"])
    assert (decision.section, decision.action) == (case["section"], case["action"])
    if case["action"] == "reply":
        write.assert_called_once_with(case["message"], case["section"])
    else:
        write.assert_not_called()
        if case["id"] not in {"unsupported", "bluetooth-basic-steps-exhausted"}:
            classify.assert_not_called()


def test_missing_knowledge(flow, monkeypatch):
    from app.ai import agent
    from app.ai.schemas import Classification

    monkeypatch.setattr(flow, "classify", lambda _: Classification(
        section="products", language="en", needs_human=False,
        reason="product question", acknowledgement="",
    ))
    lookup = Mock(side_effect=lambda name: [] if name == "products" else [{"content": "Policy"}])
    generate = Mock()
    monkeypatch.setattr(agent.db, "get_section", lookup)
    monkeypatch.setattr(agent.support_agent, "run_sync", generate)
    assert flow.decide("Does it support USB-C PD?").failure == "missing-knowledge"
    assert [call.args[0] for call in lookup.call_args_list] == [
        "brand-and-tone", "escalation-rules", "products",
    ]
    generate.assert_not_called()


@pytest.mark.parametrize(("error", "failure"), [
    (ModelHTTPError(status_code=429, model_name="groq"), "provider"),
    (groq.APIConnectionError(request=httpx.Request("POST", "https://example.test")), "provider"),
    (UnexpectedModelBehavior("bad JSON"), "structured-output"),
    (psycopg.OperationalError("test database failure"), "database"),
])
def test_dependency_failure(flow, monkeypatch, error, failure):
    from app.ai.schemas import Classification

    monkeypatch.setattr(flow, "classify", lambda _: Classification(
        section="products", language="en", needs_human=False,
        reason="product question", acknowledgement="",
    ))
    monkeypatch.setattr(flow, "write_reply", Mock(side_effect=error))
    result = flow.decide("Does the NG-C65 support PD?")
    assert result.action == "escalate"
    assert result.failure == failure
    assert result.reply == ""


@pytest.mark.parametrize("draft", [" ", "We will ship a replacement.", "Your refund has been approved."])
def test_invalid_draft(flow, monkeypatch, draft):
    from app.ai.schemas import Classification

    monkeypatch.setattr(flow, "classify", lambda _: Classification(
        section="products", language="en", needs_human=False,
        reason="product question", acknowledgement="",
    ))
    monkeypatch.setattr(flow, "write_reply", lambda *_: answer(reply=draft))
    assert flow.decide("Does the NG-C65 support PD?").action == "escalate"


def test_native_groq_request_has_schema_and_no_tools(monkeypatch):
    from app.ai import agent
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    expected = answer()
    monkeypatch.setattr(agent.db, "get_section", lambda name: [{"id": 1, "title": name, "content": "KB article"}])

    def respond(request):
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert not payload.get("tools")
        assert "KB article" in payload["messages"][-1]["content"]
        return httpx.Response(200, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "openai/gpt-oss-120b",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": expected.model_dump_json(),
            }}],
        })

    client = groq.AsyncGroq(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    model = GroqModel("openai/gpt-oss-120b", provider=GroqProvider(groq_client=client))
    with agent.support_agent.override(model=model):
        assert agent.write_reply("Earbuds do not pair.", "troubleshooting") == expected


def test_troubleshooting_loads_catalog_and_policies(monkeypatch):
    from app.ai import agent

    lookup = Mock(side_effect=lambda name: [{"id": 1, "title": name, "content": "KB article"}])
    generate = Mock(return_value=Mock(output=answer()))
    monkeypatch.setattr(agent.db, "get_section", lookup)
    monkeypatch.setattr(agent.support_agent, "run_sync", generate)

    result = agent.write_reply("My NG-BUDS2 will not pair.", "troubleshooting")

    assert result.action == "reply"
    prompt = json.loads(generate.call_args.args[0])
    assert list(prompt["company_knowledge"]) == [
        "brand-and-tone", "escalation-rules", "troubleshooting", "products",
    ]
    assert prompt["untrusted_customer_thread"] == "My NG-BUDS2 will not pair."
