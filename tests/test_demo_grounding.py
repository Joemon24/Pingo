import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evals.groundedness import Claim, Inventory
from scripts import demo_grounding


def test_label_form_uses_exact_capture_and_leaves_labels_blank(tmp_path, monkeypatch):
    report = tmp_path / "bluetooth.json"
    text = "Charge the earbuds."
    articles = [{"id": 1, "content": text}]
    report.write_text(json.dumps({
        "prompt_version": "pingo-v2-citations", "decision": {"section": "troubleshooting"},
        "generation_evidence": {"generated_draft": {"reply": text}, "articles": articles},
        "judge_decisions": ["private verdict"],
    }))
    monkeypatch.setattr(demo_grounding, "check_model", Mock())
    extractor = Mock()
    extractor.run_sync.return_value = SimpleNamespace(output=Inventory(claims=[Claim(id="c1", text=text)]))
    monkeypatch.setattr(demo_grounding, "make_agents", lambda _: (extractor, None))
    output = tmp_path / "blind"
    assert demo_grounding.label_form(report, output, "groq:fixture") == 1
    form = json.loads((output / "human_labels.template.json").read_text())
    assert form["claims"] == [{"id": "bluetooth-demo/c1", "text": text, "label": None}]
    evidence = json.loads((output / "human_evidence.json").read_text())
    assert evidence["cases"][0]["articles"] == articles
    assert "private verdict" not in json.dumps(evidence)


@pytest.mark.parametrize("second_attempt_valid", [True, False])
def test_label_form_retries_one_exact_span_failure(tmp_path, monkeypatch, second_attempt_valid):
    report = tmp_path / "bluetooth.json"
    draft = "Turn on Bluetooth, then try pairing again."
    report.write_text(json.dumps({
        "prompt_version": "pingo-v2-citations", "decision": {"section": "troubleshooting"},
        "generation_evidence": {"generated_draft": {"reply": draft}, "articles": [{"id": 1, "content": draft}]},
    }))
    invalid = Inventory(claims=[Claim(id="c1", text="Turn on Bluetooth.")])
    valid = Inventory(claims=[Claim(id="c1", text=draft)])
    extractor = Mock()
    extractor.run_sync.side_effect = [
        SimpleNamespace(output=invalid),
        SimpleNamespace(output=valid if second_attempt_valid else invalid),
    ]
    monkeypatch.setattr(demo_grounding, "check_model", Mock())
    monkeypatch.setattr(demo_grounding, "make_agents", lambda _: (extractor, None))
    output = tmp_path / "blind"

    if second_attempt_valid:
        assert demo_grounding.label_form(report, output, "groq:fixture") == 1
        form = json.loads((output / "human_labels.template.json").read_text())
        assert form["claims"] == [{"id": "bluetooth-demo/c1", "text": draft, "label": None}]
    else:
        with pytest.raises(ValueError, match="Extraction failed validation after one retry"):
            demo_grounding.label_form(report, output, "groq:fixture")
        assert not output.exists()

    assert extractor.run_sync.call_count == 2
    first, second = [json.loads(call.args[0]) for call in extractor.run_sync.call_args_list]
    assert first == {"draft": draft}
    assert second["draft"] == draft
    assert "exact contiguous span" in second["validation_feedback"]
    artifact_file = output.with_suffix(".validation-failures.json")
    failures = json.loads(artifact_file.read_text())["failures"]
    expected_attempts = ["first"] if second_attempt_valid else ["first", "after_retry"]
    assert [failure["attempt"] for failure in failures] == expected_attempts
    assert failures[0]["rejected_output"] == invalid.model_dump()
