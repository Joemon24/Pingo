# Pingo

Pingo triages support email for Gears and Co, a fictional retailer selling a 65 W USB-C charger, a 20,000 mAh power bank, and wireless Bluetooth earbuds.

## Run locally

1. Run `uv sync --locked` with Python 3.13.
2. Copy `scripts/.env.example` to `.env` and fill in your test Gmail OAuth, Supabase PostgreSQL, and Groq credentials. Keep `DRY_RUN=true`.
3. On your development database, apply all pending migrations before starting the worker. Run `uv run python -m scripts.seed_knowledge_base` to upsert the fictional articles. Run it again after editing the Markdown.
4. Run `uv run python -m app.poller` using a test inbox. Dry-run mode writes tickets and drafts to PostgreSQL and applies Gmail labels. Use `uv run uvicorn app.main:app` for the separate health endpoint.

## How decisions work

Gmail messages are claimed by unique message ID before processing. Python checks the subject and body for safety, refund, legal, sensitive-data, and suspicious instructions before involving Groq. The 20B model classifies routine requests. Python loads the chosen PostgreSQL section and company policies, plus the product catalog for troubleshooting. The 120B model returns a structured draft via Groq's native JSON schema mode. Pydantic validates the structure. Python checks the proposed action and can route the case to human review.

Drafts and execution traces are stored in `events.result`. Human review decisions are stored in `draft_reviews`. Escalations also update the ticket and apply the Gmail `needs-human` label. Event and ticket updates share a database transaction. If Gmail labeling fails after that commit, the next poll reapplies labels from the stored decision. Processing errors are capped at three attempts, with retry delays of 30 and 60 seconds before polling delay. An uncompleted claim becomes eligible for recovery after five minutes when the message is polled again. The attempt number prevents an old worker from overwriting a newer result. Exhausted events receive the Gmail `needs-human` and `ai-processed` labels and remain available for inspection in PostgreSQL. Replaying an exhausted event requires resetting its database claim and clearing the Gmail processed label after review. Delivery uses a separate explicit request after approval, with `DELIVERY_ENABLED=false` as the default.

## Test and evaluate

```sh
uv run --with pytest python -m pytest -q
```

The default suite uses mocks, including an HTTP transport test for Groq's request schema. The opt-in run below calls Groq with synthetic messages and uses Markdown fixtures for retrieval. It prints routing and model classification accuracy, safety recall, draft checks, structured-output success, and latency. Keyword checks cover the golden fixtures and require human review for broader factual accuracy.

```sh
PINGO_LIVE_GROQ=1 uv run --with pytest python -m pytest -q -s tests/test_live_groq.py
```

## Dashboard and free hosting

Set a unique `DASHBOARD_PASSWORD`, open `http://localhost:8000`, and sign in with `DASHBOARD_USERNAME` (default `admin`). Use the case queue to read inbound messages, edit drafts, and record approvals or rejections. A review is final for that draft. When delivery is explicitly enabled, the send action uses the exact stored approved text. **Check Gmail** checks one unprocessed inbox message on demand. Use a dedicated test inbox. Pingo can write database records and apply Gmail labels.

1. Use a development Supabase database. Apply the pending SQL migrations in timestamp order and seed the Markdown knowledge base locally. For IPv4-only hosts use the Supabase session pooler URL and TLS.
2. Import this repository as a Render Blueprint using `render.yaml`. Supply the test Gmail OAuth settings, Supabase connection URL, and Groq key in Render's environment settings. Retrieve the generated dashboard password there.
3. Open the deployed HTTPS URL, confirm `/health` responds, and sign in. Use **Check Gmail** with a synthetic Bluetooth email, then verify the resulting draft, review history, and dry-run Gmail labels.

The free Render service can sleep during inactivity. Use the dashboard's on-demand check for the hosted demo. For continuous polling in a test environment run the existing poller as a separate process.


## Approved delivery

Keep `DRY_RUN=true` for ingestion. `DELIVERY_ENABLED` is a separate switch, false by default. Apply the approved-delivery migration before using the dashboard.

When enabled in a test environment, **Send approved reply** claims a delivery record once. It refuses rejected, superseded, escalated, or already attempted cases. On an uncertain Gmail outcome, inspect the Gmail sent folder and reconcile the delivery record manually. Retrying uncertain delivery automatically could send a duplicate.

The API currently supports a single reviewer account through HTTP Basic authentication. Host it behind HTTPS. It does not yet provide an OAuth account-connection page, multiple reviewers, or automatic scheduling on a sleeping free host.

## Evaluation baseline and execution evidence

Run `uv run python -m evals.run --live` to evaluate the synthetic golden cases with Groq and PostgreSQL knowledge retrieval. It stores a report in `evaluation_runs` and `evals/results/latest.json`. Reports include routing accuracy, per-class precision/recall/F1, safety recall, structured stage success, latency, model names, and prompt and knowledge fingerprints. The small starter dataset needs expansion before drawing quality conclusions.

Keyword checks identify selected known failure patterns. Correctness, groundedness, and retrieval relevance remain unmeasured until human grading. Inference cost is unmeasured because no verified rate schedule is configured. No LLM-selected tools are used.

Draft execution history records stage timing, token usage, and retrieved article titles and fingerprints without copying raw model messages into telemetry. Customer messages and drafts remain sensitive database records.

## Build and CI

`docker build -t pingo .` builds the web service. Run with `docker run --rm --env-file .env -p 8000:8000 pingo`. CI runs mocked tests and the Docker build. Neither Docker nor CI has been verified in the current restricted development environment.

`/health` checks the running API. Authenticated `/ready` checks required database columns and knowledge section count.

Knowledge edits made through the dashboard have an audit record and reject stale edits. Markdown reseeding overwrites matching article contents, so export dashboard edits into Markdown before reseeding.

## Current verification limits

The audit and automated tests use installed dependencies from the Desktop environment because dependency downloads are blocked here. The workspace environment still needs a successful `uv sync --locked`. Supabase migrations, Gmail access, live Groq behavior, browser interactions, Docker build, CI execution, and public deployment require verification. The workspace is the current implementation; the Desktop source checkout is older.

## Citation integrity

New generated replies include claim-to-source records with a retrieved article ID and a verbatim supporting snippet. Python checks that the article was retrieved, that the snippet occurs in its content, and that the claim occurs in the draft. Missing or invalid citations route the case to review with an empty customer reply. Citation evidence is stored with the decision and displayed in execution history.

These checks establish citation integrity only. They do not establish that the snippet entails the claim or that every factual claim was cited. For example, a real Bluetooth quote could still be attached to an invented LED claim. Semantic evaluation and human review remain necessary. Existing drafts keep their original evidence, if any, and must be reviewed as historical outputs. The citation prompt is versioned as pingo-v2-citations. No before/after hallucination rate has been measured yet.

## Groundedness judge and calibration

The offline judge in `evals/groundedness.py` first extracts atomic claims from the draft without seeing generator citations or the knowledge base. A second call judges every extracted claim against the full generation-time article snapshots, using supported, unsupported, or contradicted labels. Python rejects fabricated judge evidence and missing/duplicate inventory judgments. Extraction completeness and semantic judgments still need human review.

New `evals.run` reports preserve the generated draft, even if citation validation later rejects it, and its actual retrieved articles in `generation_evidence`. This capture is enabled only during synthetic evaluation runs. Older reports lacking snapshots cannot be retrospectively graded reliably by looking up today's articles.

The calibration scaffold contains 15 synthetic drafts and 45 claims. It mixes supported drafts, altered facts, and misleading authentic citations. Seed labels are author-provided development labels, not independent human judgments. These cases are repetitive controls and do not establish generalization. Use `evals/human_evidence.json` and fill a copy of `evals/human_labels.template.json` before inspecting seed or judge labels. Labels are supported, unsupported, or contradicted. Do not edit the template itself. Review whether claims were extracted completely on additional natural drafts.

Both live commands require an explicit Groq judge model and check account-visible models before calling it:

```sh
uv run python -m evals.calibrate --live --judge-model groq:MODEL_ID --human-labels evals/human_labels.completed.json
uv run python -m evals.judge_report evals/results/latest.json --live --judge-model groq:MODEL_ID
```

Replace MODEL_ID with a model accessible to your Groq account. Model-family diversity is useful only after availability and judge calibration are verified. No paid provider or fallback is introduced.

Seed results measure agreement with reference annotations. Natural results use labels supplied for the sampled drafts. They report unsupported precision/recall, confusion counts, case failures, and sample size. Report judging measures extracted-claim and affected-draft rates, distinguishing unsupported from contradicted claims. Deterministic specificity flags are observation-only lexical checks, with no send-blocking effect. Calibration reports observation rates against the supplied labels.

The operator ran live seeded grading and partial extraction. A full extraction report, natural calibration, and a paired prompt comparison are pending. The original generation prompt has no committed history in this checkout. A historical baseline requires an archived prompt or an explicitly documented reconstruction. Keep input cases, article snapshots, and model settings identical across a comparison, and save each report to a distinct output filename.
