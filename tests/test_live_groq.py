import os
from time import perf_counter

import pytest

from test_support import CASES


@pytest.mark.skipif(os.getenv("PINGO_LIVE_GROQ") != "1", reason="Live Groq is opt-in")
def test_live_groq_golden(monkeypatch):
    from app.ai import decide as flow
    from app import db
    from pathlib import Path
    from scripts.seed_knowledge_base import parse

    # Use the versioned KB without contacting Gmail or Supabase.
    root = Path(__file__).parents[1] / "knowledge_base"
    monkeypatch.setattr(db, "get_section", lambda section: [
        {"id": sorted(path.stem for path in root.glob("*.md")).index(section) * 100 + index + 1, "title": title, "content": content}
        for index, (_, title, content) in enumerate(parse(root / f"{section}.md"))
    ])

    classify = flow.classify
    from app.ai.agent import support_agent

    run_reply = support_agent.run_sync
    classification_attempts = classification_matches = structured_attempts = structured_successes = 0
    expected_section = ""

    def track_classification(thread):
        nonlocal classification_attempts, classification_matches, structured_attempts, structured_successes
        classification_attempts += 1
        structured_attempts += 1
        result = classify(thread)
        structured_successes += 1
        classification_matches += result.section == expected_section
        return result

    def track_reply(prompt):
        nonlocal structured_attempts, structured_successes
        structured_attempts += 1
        result = run_reply(prompt)
        structured_successes += 1
        return result

    monkeypatch.setattr(flow, "classify", track_classification)
    monkeypatch.setattr(support_agent, "run_sync", track_reply)
    section_matches = safety_hits = unsupported = reply_cases = 0
    latencies = []
    problems = []
    for case in CASES:
        expected_section = case["section"]
        started = perf_counter()
        result = flow.decide(case["message"])
        latencies.append((perf_counter() - started) * 1000)
        section_matches += result.section == case["section"]
        safety_hits += bool(case.get("safety") and result.action == "escalate")

        if (result.section, result.action) != (case["section"], case["action"]) or result.failure:
            problems.append(case["id"])
        if result.action == "reply":
            reply_cases += 1
            text = result.reply.casefold()
            invalid = (
                case["action"] != "reply"
                or any(word not in text for word in case.get("required", []))
                or any(word in text for word in case.get("unsupported", []))
            )
            unsupported += invalid
            if invalid:
                problems.append(case["id"])

    print(f"Routing section matches {section_matches}/{len(CASES)}")
    print(f"Safety recall {safety_hits}/{sum(bool(case.get('safety')) for case in CASES)}")
    print(f"Model classification accuracy {classification_matches}/{classification_attempts}")
    print(f"Structured model stages completed {structured_successes}/{structured_attempts}")
    print(f"Invalid or unsupported drafts by fixture checks {unsupported}/{reply_cases}")
    print(f"Per-case latency in milliseconds {dict(zip((case['id'] for case in CASES), map(round, latencies)))}")
    assert not problems, problems
