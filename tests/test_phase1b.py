"""Synthetic test fixtures only. These do not supply the user's human labels."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evals.groundedness import (
    Claim, Inventory, Grade, Judgment, Evidence, build_grader_input,
    judge_draft, make_agents, smoke_agents,
)
from evals.measurement import metrics, summarize, rate, claim_covered, load_labels
from evals.calibrate import calibrate_case, run_calibration, main
from evals.sample_natural import sample_claims, write_sample

ARTICLES = [{"id": 1, "content": "Charge the earbuds."}]
INVENTORY = Inventory(claims=[Claim(id="c1", text="Charge the earbuds.")])
GRADE = Grade(judgments=[Judgment(
    claim_id="c1", label="supported", evidence=[Evidence(article_id=1, quote="Charge the earbuds.")],
    explanation="Matches the instruction.",
)])
CASE = {"id": "fixture", "kind": "supported", "draft": "Charge the earbuds.",
        "articles": ARTICLES, "claims": [{"id": "c1", "text": "Charge the earbuds."}]}


def test_calibration_and_judge_payloads_are_identical():
    extractor, grader = Mock(), Mock()
    extractor.run_sync.return_value.output = INVENTORY
    grader.run_sync.return_value.output = GRADE
    judge_draft(CASE["draft"], ARTICLES, extractor, grader)
    judge_payload = grader.run_sync.call_args.args[0]
    calibrate_case(CASE, grader)
    assert grader.run_sync.call_args.args[0] == judge_payload
    assert json.loads(judge_payload) == {
        "draft": CASE["draft"], "claims": INVENTORY.model_dump(), "articles": ARTICLES,
    }


def test_make_agents_native_output_against_installed_function_model():
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.profiles import ModelProfile

    modes = []

    def respond(messages, info):
        params = info.model_request_parameters
        modes.append(params.output_mode)
        assert params.output_mode == "native"
        assert not params.function_tools
        assert not params.output_tools
        schema = params.output_object.json_schema
        data = INVENTORY if "claims" in schema["properties"] else GRADE
        return ModelResponse(parts=[TextPart(data.model_dump_json())])

    extractor, grader = make_agents(FunctionModel(
        respond, profile=ModelProfile(supports_json_schema_output=True),
    ))
    assert judge_draft(CASE["draft"], ARTICLES, extractor, grader)["grade"] == GRADE.model_dump()
    assert modes == ["native", "native"]


def test_smoke_calls_each_agent_once_with_request_limit(capsys):
    extractor, grader = Mock(), Mock()
    extractor.run_sync.return_value.output = INVENTORY
    grader.run_sync.return_value.output = GRADE
    assert smoke_agents(extractor, grader)
    for agent in (extractor, grader):
        agent.run_sync.assert_called_once()
        assert agent.run_sync.call_args.kwargs["usage_limits"].request_limit == 1
    assert "SMOKE grader: native structured output OK" in capsys.readouterr().out


def test_smoke_prints_failure_and_still_checks_other_agent(capsys):
    extractor, grader = Mock(), Mock()
    extractor.run_sync.side_effect = TimeoutError
    grader.run_sync.return_value.output = GRADE
    assert not smoke_agents(extractor, grader)
    assert "FAILED (TimeoutError)" in capsys.readouterr().out
    grader.run_sync.assert_called_once()


def test_live_calibration_requires_human_labels_before_network(monkeypatch):
    check = Mock()
    monkeypatch.setattr("evals.calibrate.check_model", check)
    with pytest.raises(SystemExit) as exc:
        main(["--live", "--judge-model", "groq:test"])
    assert exc.value.code == 2
    check.assert_not_called()


def test_unfilled_real_template_is_rejected():
    with pytest.raises(ValueError, match="Finish all human"):
        load_labels(Path("evals/groundedness_seeds.json"), Path("evals/human_labels.template.json"))


def test_label_checks_reject_duplicate_missing_and_hash_mismatch(monkeypatch, tmp_path):
    from hashlib import sha256

    dataset = tmp_path / "fixture.json"
    labels = tmp_path / "fixture-labels.json"
    dataset.write_text(json.dumps({"cases": [CASE]}))
    blank = {"id": "fixture/c1", "label": None}
    for rows, digest, message in [
        ([blank, blank], sha256(dataset.read_bytes()).hexdigest(), "exactly once"),
        ([], sha256(dataset.read_bytes()).hexdigest(), "exactly once"),
        ([blank], "wrong", "different dataset"),
    ]:
        labels.write_text(json.dumps({"dataset_sha256": digest, "claims": rows}))
        with pytest.raises(ValueError, match=message):
            load_labels(dataset, labels)


def test_failure_accounting_intervals_and_strata():
    rows = [
        {"expected": "unsupported", "actual": "unsupported", "stratum": "seeded", "flagged": True},
        {"expected": "unsupported", "actual": None, "stratum": "natural", "flagged": False},
        {"expected": "supported", "actual": "unsupported", "stratum": "natural", "flagged": True},
        {"expected": "supported", "actual": None, "stratum": "seeded", "flagged": False},
    ]
    report = summarize(rows)
    clean = report["combined"]["excluding_failures"]
    worst = report["combined"]["failures_as_misses"]
    assert clean["recall"]["value"] == 1
    assert worst["recall"]["value"] == 0.5
    assert worst["precision"]["value"] == 0.5
    assert worst["false_negative"] == 1
    assert worst["failed_nonpositive"] == 1
    assert worst["true_negative"] == 0
    assert worst["class_counts"] == {"supported": 2, "unsupported": 2}
    assert report["natural"]["excluding_failures"]["claims"] == 1
    assert report["combined"]["specificity"]["supported_false_alarm"]["value"] == 0.5
    assert rate(0, 0)["wilson95"] is None
    assert rate(5, 10)["wilson95"] == pytest.approx([0.236593, 0.763407], abs=1e-6)


def test_failed_grading_keeps_all_reference_claims_and_types():
    grader = Mock()
    grader.run_sync.side_effect = TimeoutError
    result = run_calibration(
        [({"cases": [CASE]}, {"fixture/c1": "unsupported"})], Mock(), grader, False,
    )
    assert result["judge_failure_count"] == 1
    assert result["failure_types"] == {"grading/TimeoutError": 1}
    assert result["metrics"]["combined"]["failures_as_misses"]["false_negative"] == 1


def test_extraction_recall_rule_and_failure_denominator():
    assert claim_covered("Charge the earbuds.", "Please charge the earbuds.")
    assert not claim_covered("LED", "LEDS")
    assert not claim_covered("The LED flashes blue.", "Charge the earbuds.")
    extractor = Mock()
    extractor.run_sync.side_effect = TimeoutError
    result = run_calibration([({"cases": [CASE]}, {"fixture/c1": "supported"})], extractor, Mock(), True)
    assert result["extraction"]["seeded/supported"]["recall"]["value"] == 0
    assert result["extraction"]["seeded/supported"]["failed_drafts"] == 1


def test_natural_sampling_reproducible_balanced_and_blank(tmp_path):
    candidates = [
        {"category": category, "case_id": f"{category}-{i}", "claim_id": "c1",
         "draft": CASE["draft"], "text": CASE["draft"], "articles": ARTICLES}
        for category in ("products", "troubleshooting") for i in range(20)
    ]
    selected = sample_claims(candidates, 30, 42)
    assert selected == sample_claims(candidates, 30, 42)
    assert sum(row["category"] == "products" for row in selected) == 15
    assert len({row["case_id"] for row in selected}) == 30
    write_sample(selected, tmp_path / "sample", "report-hash", 42, [])
    form = json.loads((tmp_path / "sample/human_labels.template.json").read_text())
    assert len(form["claims"]) == 30
    assert all(row["label"] is None for row in form["claims"])
    with pytest.raises(ValueError):
        sample_claims([], 30, 42)


def test_prompt_archive_exact_reconstruction():
    from app.ai.prompts import load_prompt

    root = Path("prompts")
    v2 = load_prompt("pingo-v2-citations")
    citation_text = json.loads((root / "citation-instructions.json").read_text())
    assert v2.count(citation_text) == 1
    assert v2.replace(citation_text, "") == load_prompt("pingo-v1-reconstructed")
    with pytest.raises(ValueError):
        load_prompt("../.env")


def test_label_provenance_metadata_uses_supplied_bytes(monkeypatch, tmp_path):
    # Synthetic fixture labels only; never touch user calibration labels.
    from hashlib import sha256

    dataset = tmp_path / "fixture.json"
    labels = tmp_path / "fixture-labels.json"
    dataset.write_text(json.dumps({"cases": [CASE]}))
    labels.write_text(json.dumps({
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "claims": [{"id": "fixture/c1", "label": "supported"}],
    }))
    _, _, metadata = load_labels(dataset, labels)
    assert metadata["labels_sha256"] == sha256(labels.read_bytes()).hexdigest()
    assert metadata["labels_mtime_ns"] == labels.stat().st_mtime_ns


def test_cli_prints_failures_first_and_reports_provenance(monkeypatch, tmp_path, capsys):
    grader, extractor = Mock(), Mock()
    grader.run_sync.side_effect = TimeoutError
    monkeypatch.setattr("evals.calibrate.load_labels", lambda *_: (
        {"cases": [CASE]}, {"fixture/c1": "unsupported"},
        {"labels_sha256": "fixture-hash", "labels_mtime_ns": 123},
    ))
    monkeypatch.setattr("evals.calibrate.check_model", Mock())
    monkeypatch.setattr("evals.calibrate.make_agents", lambda *_: (extractor, grader))
    output = tmp_path / "report.json"
    main(["--live", "--judge-model", "groq:fixture", "--human-labels", "fixture-labels.json",
          "--output", str(output)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["Judge failures: 1", 'Failure types: {"grading/TimeoutError": 1}']
    saved = json.loads(output.read_text())
    assert saved["inputs"][0]["labels_sha256"] == "fixture-hash"
    assert saved["metrics"]["combined"]["failures_as_misses"]["false_negative"] == 1


def test_mixed_strata_calibration_uses_human_labels_without_seed_annotations():
    grader = Mock()
    grader.run_sync.return_value.output = GRADE
    grader.run_sync.return_value.usage = SimpleNamespace(input_tokens=10, output_tokens=5)
    inputs = [
        ({"stratum": stratum, "cases": [CASE]}, {"fixture/c1": "supported"})
        for stratum in ("seeded", "natural")
    ]
    report = run_calibration(inputs, Mock(), grader, False)
    assert report["metrics"]["combined"]["excluding_failures"]["claims"] == 2
    assert report["metrics"]["seeded"]["excluding_failures"]["claims"] == 1
    assert report["metrics"]["natural"]["excluding_failures"]["claims"] == 1


def test_extraction_candidates_preserve_snapshots_without_labels():
    from evals.sample_natural import extract_candidates

    extractor = Mock()
    extractor.run_sync.return_value.output = INVENTORY
    candidates, failures = extract_candidates({"results": [{
        "id": "natural-fixture", "expected_section": "troubleshooting",
        "generation_evidence": {"articles": ARTICLES, "generated_draft": {"reply": CASE["draft"]}},
    }]}, extractor)
    assert not failures
    assert candidates[0]["articles"] == ARTICLES
    assert candidates[0]["category"] == "troubleshooting"
    assert "label" not in candidates[0]


def test_smoke_cannot_accidentally_start_calibration(monkeypatch):
    check = Mock()
    monkeypatch.setattr("evals.calibrate.check_model", check)
    with pytest.raises(SystemExit):
        main(["--live", "--judge-model", "groq:fixture", "--smoke", "--extraction"])
    check.assert_not_called()


def test_rate_limit_retries_once_then_defers_other_calls(monkeypatch):
    from pydantic_ai.exceptions import ModelRateLimitError

    pause = Mock()
    monkeypatch.setattr("evals.calibrate.sleep", pause)
    grader = Mock()
    grader.run_sync.side_effect = ModelRateLimitError("fixture", "rate limited", retry_after=2)
    dataset = {"cases": [CASE, {**CASE, "id": "later"}]}
    report = run_calibration(
        [(dataset, {"fixture/c1": "supported", "later/c1": "unsupported"})],
        Mock(), grader, False,
    )
    pause.assert_called_once_with(2)
    assert grader.run_sync.call_count == 2
    assert report["judge_failure_count"] == 1
    assert report["deferred_cases"] == ["seeded/later"]
    assert report["metrics"]["combined"]["failures_as_misses"]["false_negative"] == 1


def test_rate_limit_recovery_keeps_extraction_result(monkeypatch):
    from pydantic_ai.exceptions import ModelRateLimitError

    pause = Mock()
    monkeypatch.setattr("evals.calibrate.sleep", pause)
    extractor = Mock()
    extractor.run_sync.side_effect = [
        ModelRateLimitError("fixture", "rate limited"), SimpleNamespace(output=INVENTORY),
    ]
    report = run_calibration(
        [({"cases": [CASE]}, {"fixture/c1": "supported"})], extractor, Mock(), True,
    )
    pause.assert_called_once_with(60)
    assert report["failures"] == []
    assert report["extraction"]["seeded/supported"]["covered"] == 1


def test_pacing_skips_cached_cases_and_spaces_new_calls_and_retries(monkeypatch):
    from unittest.mock import call
    from pydantic_ai.exceptions import ModelRateLimitError

    pause = Mock()
    monkeypatch.setattr("evals.calibrate.sleep", pause)
    extractor = Mock()
    output = SimpleNamespace(output=INVENTORY)
    extractor.run_sync.side_effect = [
        output, ModelRateLimitError("fixture", "rate limited", retry_after=1), output,
    ]
    dataset = {"cases": [CASE, {**CASE, "id": "second"}, {**CASE, "id": "third"}]}
    labels = {f"{case['id']}/c1": "supported" for case in dataset["cases"]}
    report = run_calibration(
        [(dataset, labels)], extractor, Mock(), True,
        previous_results={"seeded/fixture": {"case_id": "seeded/fixture", "inventory": INVENTORY.model_dump()}},
        interval_seconds=25,
    )
    assert extractor.run_sync.call_count == 3
    assert pause.call_args_list == [call(25), call(25)]
    assert report["extraction"]["seeded/supported"]["covered"] == 3


def test_calibration_reports_invalid_judge_evidence_with_fixed_reason(capsys):
    grader = Mock()
    invalid = GRADE.model_copy(deep=True)
    invalid.judgments[0].evidence[0].quote = "Provider output that must stay out of the error"
    grader.run_sync.return_value.output = invalid
    report = run_calibration(
        [({"cases": [CASE]}, {"fixture/c1": "supported"})], Mock(), grader, False,
    )
    assert report["failures"] == [{
        "case_id": "seeded/fixture", "stage": "grading",
        "error_type": "JudgeOutputError", "validation_issue": "Judge cited an unavailable quote",
    }]
    error_output = capsys.readouterr().err
    assert "seeded/fixture: Judge cited an unavailable quote" in error_output
    assert invalid.judgments[0].evidence[0].quote not in error_output


@pytest.mark.parametrize("extraction", [False, True])
def test_resume_keeps_successes_and_only_runs_missing_cases(extraction):
    from evals.calibrate import reused_results

    inputs = [({"cases": [CASE, {**CASE, "id": "later"}]},
               {"fixture/c1": "supported", "later/c1": "supported"})]
    mode = "extraction" if extraction else "fixed_inventory_grading"
    field, cached = ("inventory", INVENTORY) if extraction else ("grade", GRADE)
    report = {
        "mode": mode, "judge_model": "groq:fixture",
        "failures": [{"case_id": "seeded/later"}],
        "results": [{"case_id": "seeded/fixture", field: cached.model_dump()},
                    {"case_id": "seeded/later"}],
    }
    resumed = reused_results(report, {"mode": mode, "judge_model": "groq:fixture"}, inputs)
    extractor, grader = Mock(), Mock()
    extractor.run_sync.return_value.output = INVENTORY
    grader.run_sync.return_value = SimpleNamespace(
        output=GRADE, usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    result = run_calibration(inputs, extractor, grader, extraction, previous_results=resumed)
    assert result["reused_case_count"] == 1
    assert result["failures"] == []
    assert extractor.run_sync.call_count == int(extraction)
    assert grader.run_sync.call_count == int(not extraction)
    if extraction:
        assert result["extraction"]["seeded/supported"]["covered"] == 2
    else:
        assert result["metrics"]["combined"]["excluding_failures"]["agreement"]["numerator"] == 2
    with pytest.raises(ValueError, match="different judge_model"):
        reused_results(report, {"mode": mode, "judge_model": "groq:other"}, inputs)
