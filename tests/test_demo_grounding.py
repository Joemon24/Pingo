import json
from types import SimpleNamespace
from unittest.mock import Mock

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
