# Deploy the draft-only Pingo demo

One Docker web service serves the dashboard and API. Supabase holds tickets, drafts,
reviews, and knowledge. Render's environment holds OAuth credentials. Use a **dedicated
demo Gmail account** and synthetic support messages. Keep `DRY_RUN=true` and
`DELIVERY_ENABLED=false` in every environment used below.

Render supports a Docker web service on **Free compute**. After 15 minutes without
inbound traffic it spins down, and the next request can take about a minute to wake it.
The container filesystem is ephemeral. The demo polls one inbox message per **Check Gmail**
click and provisions only the web service. The existing worker entry point is
`.venv/bin/python -m app.poller`; leave it stopped for the thin demo.

[Render Free limits](https://render.com/docs/free) · [Docker web services](https://render.com/hosting/docker) · [Blueprint reference](https://render.com/docs/blueprint-spec)

## 1. Prepare Supabase and test locally

Copy the environment template into a new checkout if necessary, then edit it privately.

```sh
uv sync --locked
test -f .env || cp scripts/.env.example .env
chmod 600 .env
```

In your demo Supabase project, open **Connect → Session pooler** and use the
PostgreSQL connection URI with your database password and `sslmode=require`.
Put it in `DATABASE_URL`. Use the same database on Render.

- For a **fresh** demo project, paste and run each of the five files in
  `supabase/migrations/` in timestamp order in the Supabase SQL Editor.
- For an **existing** project, apply only migrations the database is missing.
  If you use the CLI, start with `npx supabase migration list --linked`. SQL Editor
  runs can be absent from this history. Check what is present before applying more.
  Keep the existing data and do not rerun the table-creation migration.

Seed once. If the database already has manually edited demo articles, check them first
because seeding upserts article content.

```sh
uv run --no-sync python -m scripts.seed_knowledge_base
PINGO_LIVE_GROQ=0 uv run --no-sync --with pytest python -m pytest -q
```

Expected output is `Upserted … knowledge entries` and a passing pytest summary
with one opt-in live test skipped. For a Docker check on your machine, start Docker
Desktop and run:

```sh
docker build -t pingo:m1 .
docker run --rm --env-file .env -e PORT=8000 -e DELIVERY_ENABLED=false -p 127.0.0.1:8000:8000 pingo:m1
```

Docker's env-file format uses plain `KEY=value` lines. Remove shell quoting around
older generated token values in the private env file before passing it to Docker.
In a second terminal, run the health and readiness checks in step 3 with
`BASE_URL=http://127.0.0.1:8000`.

## 2. Configure Google and Render

1. In Google Cloud, enable **Gmail API**. Under Google Auth Platform, configure the
   external consent screen and add the dedicated Gmail address as a test user.
   Request the `https://www.googleapis.com/auth/gmail.modify` scope. Create an OAuth
   client of type **Web application**. Keep the ID and secret in your private local
   `.env` and Render environment. A refresh token is tied to its OAuth client.
2. Publish the reviewed commits to a GitHub repository you own. If the checkout has
   no remote, create the empty GitHub repository first, then run:

   ```sh
   export GITHUB_REPO_URL='https://github.com/YOUR_ACCOUNT/YOUR_REPO.git'
   git remote add origin "$GITHUB_REPO_URL"
   git push -u origin HEAD
   ```

   For an existing remote, check `git remote -v` and publish the reviewed branch there.
   In GitHub **Actions → Pingo checks**, confirm the test and image-build steps for
   the commit Render will deploy.
3. In Render choose **New → Blueprint**, select that repository and branch, and
   review `render.yaml`. Confirm **Free** compute and apply. Fill the table below.
   If Render asks for a token before the callback URL exists, use
   `pending-authorization` temporarily. Keep delivery disabled. After the first
   deploy, copy the exact HTTPS web-service URL shown in Render.

| Variable | Purpose | Secret |
| --- | --- | --- |
| `DATABASE_URL` | TLS Supabase PostgreSQL session-pooler connection | Yes |
| `GROQ_API_KEY` | Groq classifier and reply calls | Yes |
| `GOOGLE_CLIENT_ID` | ID of the Google OAuth web client | No |
| `GOOGLE_CLIENT_SECRET` | Google OAuth web client secret | Yes |
| `GOOGLE_REFRESH_TOKEN` | Durable Gmail credential stored as a Render secret | Yes |
| `GOOGLE_REDIRECT_URI` | Exact hosted HTTPS callback URL | No |
| `DASHBOARD_USERNAME` | Single reviewer login, default `admin` | No |
| `DASHBOARD_PASSWORD` | Generated unique login password | Yes |
| `DRY_RUN` | Set to `true` for intake, drafts, and labels | No |
| `DELIVERY_ENABLED` | Set to `false` to refuse all delivery requests | No |
| `PORT` | Provided by Render, read by the container | No |
| `CLASSIFIER_MODEL` | Optional override, defaults to `groq:openai/gpt-oss-20b` | No |
| `AGENT_MODEL` | Optional override, defaults to `groq:openai/gpt-oss-120b` | No |
| `POLL_INTERVAL_SECONDS` | Optional for the separate worker only, default 60 | No |

4. Set the final callback URL using the exact service URL. For example, if Render
   assigned `https://pingo-support.onrender.com`, register
   **`https://pingo-support.onrender.com/oauth/gmail/callback`** as an
   **Authorized redirect URI** on the Google Web application client. Use your actual
   assigned hostname, including any suffix, with no trailing slash.
   Set this exact URI as `GOOGLE_REDIRECT_URI` in Render and save.

   ```sh
   export BASE_URL='https://YOUR-ACTUAL-SERVICE.onrender.com'
   export GOOGLE_REDIRECT_URI="$BASE_URL/oauth/gmail/callback"
   curl -fsS "$GOOGLE_REDIRECT_URI"
   uv run --no-sync python -m scripts.get_refresh_token --hosted
   ```

   The callback returns a short instruction. In the consent browser select the
   dedicated account. Paste the redirected address into the script's hidden prompt.
   The local script checks state and completes the code exchange with PKCE. It
   writes the refresh token to the private local `.env`. Copy just that token
   privately into Render's `GOOGLE_REFRESH_TOKEN` secret and redeploy. Clear the
   clipboard and close the callback tab. Never include the redirect address after
   consent, the code, or tokens in screenshots or logs. The image's Uvicorn access
   log is disabled so it does not record OAuth callback query strings.

Google external apps in **Testing** receive refresh tokens that expire after seven
days for Gmail scopes. On expiry, repeat the local authorization step and update the
Render secret. A filesystem-only token would be lost on a Render restart or redeploy.
The previous local Desktop OAuth command remains available without `--hosted`.

[Google token expiration](https://developers.google.com/identity/protocols/oauth2#expiration) · [Google redirect URI rules](https://developers.google.com/identity/protocols/oauth2/web-server)

## 3. Run the post-deploy smoke check

```sh
curl -i --max-time 120 "$BASE_URL/health"
curl -i -u admin --max-time 120 "$BASE_URL/ready"
```

Enter the dashboard password at curl's prompt. Expect HTTP 200 with
`{"status":"ok"}` and then HTTP 200 with
`{"status":"ready","delivery_enabled":false}`. Health is public and does no
database work. Readiness requires login and checks config presence, database schema,
and seeded knowledge. It does not contact Google or Groq.

1. Open `$BASE_URL/` in a private window. Sign in, check that the case list loads,
   and confirm the sidebar shows **Delivery disabled**.
2. From a separate email account, send a fresh synthetic Bluetooth pairing request
   to the dedicated demo inbox. Click **Check Gmail**, then open the matching case.
   Confirm the draft, citations, and execution history are visible. If another inbox
   message is picked up first, continue with **Check Gmail** until the demo email
   appears. Use a second independently threaded email to exercise rejection.
3. Approve one draft and reject the other. In Render select **Manual Deploy →
   Deploy latest commit**. Reopen both cases and verify their review statuses and
   approved text survived. No send control should be visible.
4. Set `CASE_ID` to an approved demo case ID (available in the authenticated queue
   response). With delivery still disabled, verify refusal:

   ```sh
   export CASE_ID='REPLACE_WITH_NUMERIC_APPROVED_CASE_ID'
   curl -i -u admin -X POST -H 'X-Pingo-Action: dashboard' \
     "$BASE_URL/api/queue/$CASE_ID/send"
   ```

   Expect HTTP 409 and `{"detail":"Delivery is disabled"}`.
5. When ready to reauthorize the dedicated Gmail account, revoke this app's access
   in the demo account's Google Account third-party connections settings. Click
   **Check Gmail** and verify **Gmail connection expired, reauthorize**. Reauthorize
   using the command above, update the Render secret, redeploy, and verify inbox
   check recovers.

The five-attempt login lockout is shared by clients in one process. A bad client can
briefly lock out the reviewer, and restart clears the counter. Use the public HTTPS
URL with a strong unique password and only demo data.

Paste back the public URL, the health/ready/refusal status lines and bodies, the
GitHub Actions run URL, and the result of the ingest, review-persistence, banner,
and reauthorization checks. Leave credentials, callback query strings, and email
addresses out.
