# Phase 1B verification contract

This phase hardens evaluation tooling. It does not initiate live measurement or change delivery settings.

## Grader parity

Both fixed-inventory calibration and extracted-claim judging use build_grader_input.
Include the full draft for pronouns, qualifications, and local context; article snapshots remain the only evidence.
Context can bias the judge toward the generator's wording, so report calibration and prompt versions.

The payload keys are draft, claims (with a claims array), and articles. Neither path supplies generator citations or seed/human labels to the model.

## Live diagnostics and human labels

Select an account-visible groq:<model ID> with --judge-model. models.list() is checked before inference. No family is hardcoded or automatically substituted. If only the generator's family is available, use a different judging prompt at temperature 0. Same-family and same-model judges can share systematic errors; temperature 0 does not remove that risk. Calibrate against independent human labels and record the exact model ID.

A standalone --smoke makes at most one request per agent with native structured output and prints a result for each. It does not require labels because it is not calibration. Calibration (including --extraction) requires one completed --human-labels file per --dataset, loaded and validated before any network call. Duplicate/missing claim IDs, null labels, and wrong dataset hashes are rejected. Reports record label-file SHA-256 and mtime in nanoseconds. No code supplies or infers human labels.

Expected smoke output on success:

    SMOKE extractor: native structured output OK
    SMOKE grader: native structured output OK

A failed agent prints FAILED plus its exception type and exits nonzero after attempting both. Model-list failure prevents either request. Native structured mode is also exercised offline through the installed Pydantic AI FunctionModel.

## Calibration accounting

Reports start with judge-failure count and failure types. Per seeded, natural, and combined stratum:
- Excluding failures includes only successfully judged cases.
- Failures-as-misses includes all human-positive unsupported claims in failed cases as false negatives. Failed non-positive claims remain unclassified, not true negatives. Precision is unchanged by missing predictions; recall and agreement decrease.
- Raw TP/FP/FN/TN, class counts, prediction counts, failed non-positive counts, and sample sizes are retained.
- Precision, recall, agreement, and specificity rates include numerator, denominator, and Wilson 95% intervals. Zero denominators produce null values and null intervals.
- Unsupported is the positive class. Contradicted remains distinct and is negative for this binary metric.
- Claims within a draft are correlated. Wilson intervals assume independent observations and are descriptive, not a population guarantee.

Specificity flags are evaluated per human-labeled claim against that draft's full articles. Report supported false-alarm and unsupported-or-contradicted hit rates. These lexical flags remain observation-only and do not route cases or authorize delivery.

## Extraction coverage

--extraction calls the extractor only. Validate exact draft spans, then compare against the reference inventory. A reference is covered if one extracted claim contains its casefolded word-token span, or shares >=90% of its word-token multiset. Extra words are allowed. This measures span recall, not atomicity, truth, or entailment. Failed extraction contributes zero covered claims with its full reference denominator. Counts are grouped by stratum and seed kind, including failed drafts.

For natural samples, reference claims came from extraction. Re-extraction recall measures consistency on that sample; it cannot establish coverage of the entire natural draft. Review omissions manually.

## Natural sample

sample_natural reads a saved live evaluation report and its generation-time article snapshots. It extracts each available generated draft once and samples 30 claims without replacement using seed 42. Allocate evenly across scenario categories (expected_section, falling back to section); redistribute exhausted categories. If fewer than 30 claims are available, fail rather than silently change N. Fixed seeding stabilizes sampling given the same extractor output; model extraction itself can vary.

The output directory contains dataset.json, human_evidence.json, and human_labels.template.json. All labels are null. Extraction failures and source report hash are retained. Calibration accepts paired repeated --dataset and --human-labels arguments for seeded and natural strata. Label independently before viewing predictions.

## Prompt archive

prompts/pingo-v2-citations.txt is the exact concatenated generation instruction text archived from the running implementation. Runtime loads this file by default.
prompts/pingo-v1-reconstructed.txt is a reconstruction, not an archived historical original. It is v2 with only the exact string stored in citation-instructions.json removed. The structured output schema, model settings, knowledge retrieval, and other policies remain the same.
evals.run accepts --prompt-version pingo-v2-citations or pingo-v1-reconstructed. The reconstructed run bypasses the citation-integrity gate to retain comparable uncited drafts. This switch is evaluation-process-local; the dashboard keeps v2.
No before/after comparison has been performed in Phase 1B.

## Raw evidence policy

evals/results/ is ignored in Git, including live drafts, judge reports, natural samples, and human label forms. Completed labels outside it matching evals/human_labels.completed*.json are ignored too. Store raw evidence there. Only a manually reviewed final baseline summary (for example docs/baseline-summary.md) should be committed after measurement; no placeholder metrics are provided.

## Local commands

Run from the Pingo checkout, after selecting an actual accessible judge ID:

    uv run --no-sync --with pytest python -m pytest -q
    export PINGO_JUDGE_MODEL='groq:CHOSEN_MODEL_ID'
    uv run --no-sync python -m evals.calibrate --live --judge-model "$PINGO_JUDGE_MODEL" --smoke

Copy evals/human_labels.template.json to evals/results/seeded-labels.json, then fill every label yourself using evals/human_evidence.json before predictions.

    uv run --no-sync python -m evals.calibrate --live --judge-model "$PINGO_JUDGE_MODEL" --human-labels evals/results/seeded-labels.json --output evals/results/seeded-calibration.json
    uv run --no-sync python -m evals.calibrate --live --judge-model "$PINGO_JUDGE_MODEL" --human-labels evals/results/seeded-labels.json --extraction --output evals/results/seeded-extraction.json

Given an existing saved live report with generation_evidence snapshots:

    uv run --no-sync python -m evals.sample_natural evals/results/existing-live-report.json --live --judge-model "$PINGO_JUDGE_MODEL" --output-dir evals/results/natural

Fill a copy of natural/human_labels.template.json at natural/human_labels.completed.json yourself. Then:

    uv run --no-sync python -m evals.calibrate --live --judge-model "$PINGO_JUDGE_MODEL" --dataset evals/groundedness_seeds.json --human-labels evals/results/seeded-labels.json --dataset evals/results/natural/dataset.json --human-labels evals/results/natural/human_labels.completed.json --output evals/results/combined-calibration.json

A calibration with complete model success prints Judge failures: 0, Failure types: {}, then measured JSON. Values depend on the actual run. Output reports and sample directories must be new, to preserve earlier evidence.

## Resume after a rate limit

Calibration retries a rate-limited case once using the provider's retry-after value, with a one-second minimum. When the provider gives no delay, calibration waits 60 seconds. A second rate-limit error, a requested delay above 90 seconds, or an explicit quota error defers the remaining calls and saves the partial report. The summary lists deferred cases separately from attempted failures. Pending claims still count in the full-set denominators.

Run the same calibration or extraction command with `--resume PATH_TO_PARTIAL_REPORT` and a fresh `--output` path. The mode, model, prompt versions and hashes, dataset hashes, and label hashes must match. A resumed report combines validated successes from the earlier run with new results and records the earlier report's hash. If another run is deferred, resume from the newest report to retain the accumulated successes.
