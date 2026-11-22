import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from evals.groundedness import (
    Claim, Inventory, Grade, Judgment, Evidence, judge_draft,
    unsupported_metrics, validate_grade, validate_inventory, specificity_flags,
)

ARTICLES = [{"id": 1, "content": "Charge the earbuds. Returns within 30 days."}]
INVENTORY = Inventory(claims=[Claim(id="c1", text="The LED flashes blue.")])


def test_judge_receives_full_articles_and_independent_inventory():
    extractor = Mock()
    extractor.run_sync.return_value.output = INVENTORY
    grader = Mock()
    grader.run_sync.return_value.output = Grade(judgments=[
        Judgment(claim_id="c1", label="unsupported", evidence=[], explanation="No LED behavior documented."),
    ])
    result = judge_draft("The LED flashes blue.", ARTICLES, extractor, grader)
    assert json.loads(extractor.run_sync.call_args.args[0]) == {"draft": "The LED flashes blue."}
    payload = json.loads(grader.run_sync.call_args.args[0])
    assert payload["articles"] == ARTICLES
    assert "generator_citations" not in payload
    assert result["grade"]["judgments"][0]["label"] == "unsupported"


@pytest.mark.parametrize("judgments", [
    [],
    [Judgment(claim_id="other", label="unsupported", evidence=[], explanation="Absent")],
    [Judgment(claim_id="c1", label="supported", evidence=[], explanation="Unsupported assertion")],
    [Judgment(claim_id="c1", label="supported", evidence=[
        Evidence(article_id=1, quote="LED flashes")], explanation="Fabricated quote")],
])
def test_missing_or_invalid_judge_evidence_is_rejected(judgments):
    with pytest.raises(ValueError):
        validate_grade(INVENTORY, Grade(judgments=judgments), ARTICLES)


def test_extractors_cannot_invent_text_or_duplicate_ids():
    with pytest.raises(ValueError):
        validate_inventory("Different draft", INVENTORY)
    with pytest.raises(ValueError):
        validate_inventory("The LED flashes blue.", Inventory(claims=INVENTORY.claims * 2))


def test_metrics_report_counts_and_unmeasured_denominators():
    result = unsupported_metrics(
        ["unsupported", "supported", "unsupported", "contradicted"],
        ["unsupported", "unsupported", "supported", "contradicted"],
    )
    assert result["true_positive"] == result["false_positive"] == result["false_negative"] == 1
    assert result["unsupported_precision"] == result["unsupported_recall"] == 0.5
    assert result["agreement_count"] == 2
    assert unsupported_metrics([], [])["agreement"] is None
    with pytest.raises(ValueError):
        unsupported_metrics(["supported"], [])


def test_specificity_backstop_is_observation_only():
    assert specificity_flags("Hold the button for 90 days; LED flashing blue.", ARTICLES) == [
        "90", "blue", "button", "flashing", "hold", "led",
    ]


def test_seed_dataset_and_blind_form_align():
    root = Path(__file__).parents[1] / "evals"
    data = json.loads((root / "groundedness_seeds.json").read_text())
    form = json.loads((root / "human_labels.template.json").read_text())
    blind = json.loads((root / "human_evidence.json").read_text())
    assert len(data["cases"]) == len(blind["cases"]) == 15
    assert len(form["claims"]) == 45
    assert all(row["label"] is None for row in form["claims"])
    assert "seed_label" not in json.dumps(blind)
    for case in data["cases"]:
        validate_inventory(case["draft"], Inventory(claims=[
            Claim(id=claim["id"], text=claim["text"]) for claim in case["claims"]
        ]))
