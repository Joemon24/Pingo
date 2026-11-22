from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evals.groundedness import Grade, Inventory, validate_grade, JudgeOutputError
from evals.triage_failure import classify_artifact, has_formatting_match, repeat_case

CASE = {
    "id": "fixture",
    "draft": "The warranty lasts 12 months.",
    "claims": [{"id": "c1", "text": "The warranty lasts 12 months."}],
    "articles": [{"id": 1, "content": "The warranty lasts 12 months."},
                 {"id": 2, "content": "Accessories are sold separately."}],
}
BASE = {
    "draft": CASE["draft"], "inventory": CASE["claims"], "stage": "grading",
    "articles": CASE["articles"], "cited_article_id": 1,
}


@pytest.mark.parametrize(("quote", "record_changes", "seed_changes", "category"), [
    ("The warranty lasts 24 months.", {}, {}, "fabricated quote"),
    ("**The warranty** lasts\n１２ months!", {}, {}, "near-verbatim quote"),
    ("Accessories are sold separately.", {}, {}, "correct quote, wrong article ID"),
    ("A quote.", {"articles": [{"id": 9, "content": "different"}]}, {}, "snapshot/ID mismatch"),
    ("A quote.", {}, {"claims": [{"id": "c1", "text": "Outside this draft."}]}, "seed-data bug"),
])
def test_classification_rules(quote, record_changes, seed_changes, category):
    assert classify_artifact({**BASE, "rejected_quote": quote, **record_changes}, {**CASE, **seed_changes}) == category


def test_number_and_semantic_differences_do_not_get_formatting_match():
    assert not has_formatting_match("Warranty lasts 24 months", "Warranty lasts 12 months")
    assert not has_formatting_match("Duration is 1.2 years", "Duration is 12 years")
    assert not has_formatting_match("Accessories are sold separately", CASE["articles"][0]["content"])


@pytest.mark.parametrize(("quote", "article_id"), [
    ("Warranty lasts 24 months.", 1),
    ("Accessories are sold separately.", 1),
    ("The warranty lasts\n12 months.", 1),
])
def test_triage_cannot_relax_the_strict_runtime_validator(quote, article_id):
    grade = Grade(judgments=[{
        "claim_id": "c1", "label": "supported", "explanation": "fixture",
        "evidence": [{"article_id": article_id, "quote": quote}],
    }])
    with pytest.raises(JudgeOutputError):
        validate_grade(Inventory(claims=CASE["claims"]), grade, CASE["articles"])


def test_repeat_case_reports_strict_failures_with_denominator():
    valid = Grade(judgments=[{
        "claim_id": "c1", "label": "supported", "explanation": "Matches source",
        "evidence": [{"article_id": 1, "quote": CASE["draft"]}],
    }])
    invalid = valid.model_copy(deep=True)
    invalid.judgments[0].evidence[0].quote = "Made up."
    grader = Mock()
    grader.run_sync.side_effect = [SimpleNamespace(output=invalid), SimpleNamespace(output=valid)]
    report, records = repeat_case(CASE, grader, 2, 0)
    assert report["validation_failures"] == {"strict": 1, "normalized": None}
    assert report["validation_rates"]["first"]["numerator"] == 1
    assert report["validation_rates"]["first"]["denominator"] == 2
    assert report["validation_rates"]["first"]["wilson95"]
    assert records[0]["repeat"] == 1
