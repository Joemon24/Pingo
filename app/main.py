import logging
import os
import secrets
from collections import deque
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from google.auth.exceptions import RefreshError, TransportError
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field

from app import reviews
from app.config import settings

app = FastAPI(title="Pingo", docs_url=None, redoc_url=None, openapi_url=None)
static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static), name="static")
basic = HTTPBasic()
poll_lock = Lock()
login_lock = Lock()
failed_logins: deque[float] = deque()
LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60
logger = logging.getLogger(__name__)


def reviewer(credentials: Annotated[HTTPBasicCredentials, Depends(basic)]) -> str:
    with login_lock:
        now = monotonic()
        while failed_logins and failed_logins[0] <= now - LOGIN_WINDOW_SECONDS:
            failed_logins.popleft()
        # One reviewer shares this bounded window across clients and request paths.
        if len(failed_logins) >= LOGIN_ATTEMPTS:
            raise HTTPException(429, "Too many login attempts. Try again in a minute.",
                                headers={"Retry-After": str(LOGIN_WINDOW_SECONDS)})
        valid_user = secrets.compare_digest(credentials.username.encode(), settings.dashboard_username.encode())
        valid_password = secrets.compare_digest(credentials.password.encode(), settings.dashboard_password.encode())
        if not settings.dashboard_password or not (valid_user and valid_password):
            failed_logins.append(now)
            raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": 'Basic realm="Pingo"'})
    return credentials.username


Reviewer = Annotated[str, Depends(reviewer)]


def require_dashboard_request(x_pingo_action: Annotated[str | None, Header()] = None) -> None:
    if x_pingo_action != "dashboard":
        raise HTTPException(403, "Dashboard request header required")


class ReviewInput(BaseModel):
    action: Literal["approve", "reject"]
    draft: str = Field(max_length=6000)


@app.middleware("http")
async def response_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response


@app.exception_handler(RequestValidationError)
def invalid_request(_request: Request, _exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default validation response includes submitted input values.
    return JSONResponse({"detail": "Invalid request data"}, status_code=422)


@app.exception_handler(psycopg.Error)
def database_unavailable(_request: Request, exc: psycopg.Error) -> JSONResponse:
    logger.warning("Dashboard database request failed (%s)", type(exc).__name__)
    return JSONResponse({"detail": "Database unavailable. Check the connection and migrations."}, status_code=503)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/oauth/gmail/callback", response_class=PlainTextResponse)
def gmail_oauth_callback() -> PlainTextResponse:
    if not settings.google_redirect_uri:
        raise HTTPException(404, "Hosted Gmail authorization is disabled")
    # The initiating local process owns the state, PKCE verifier, and token exchange.
    return PlainTextResponse(
        "Return to your local Pingo authorization command and follow its instructions.",
        headers={"Referrer-Policy": "no-referrer"},
    )


@app.get("/")
def dashboard(_: Reviewer) -> FileResponse:
    return FileResponse(static / "index.html")


@app.get("/api/status")
def status(_: Reviewer) -> dict:
    return {"delivery_enabled": settings.delivery_enabled}


@app.get("/api/queue")
def queue(_: Reviewer) -> list[dict]:
    return reviews.list_queue()


@app.get("/api/queue/{event_id}")
def case(event_id: int, _: Reviewer) -> dict:
    entry = reviews.get_case(event_id)
    if entry is None:
        raise HTTPException(404, "Case not found")
    return entry


@app.post("/api/queue/{event_id}/review", dependencies=[Depends(require_dashboard_request)])
def review(event_id: int, payload: ReviewInput, username: Reviewer) -> dict:
    draft = payload.draft.strip()
    if payload.action == "approve" and not draft:
        raise HTTPException(422, "An approved draft needs reply text")
    if not reviews.save_review(event_id, username, payload.action, draft):
        raise HTTPException(409, "Draft is already reviewed or unavailable")
    return {"review": payload.action}


@app.post("/api/poll", dependencies=[Depends(require_dashboard_request)])
def poll(_: Reviewer) -> dict:
    if not poll_lock.acquire(blocking=False):
        raise HTTPException(409, "An inbox check is already running")
    try:
        from app.poller import poll_once

        return poll_once(max_results=1)
    except RefreshError as exc:
        logger.warning("Inbox check failed: Google OAuth refresh rejected")
        if any(isinstance(arg, dict) and arg.get("error") == "invalid_grant" for arg in exc.args):
            return JSONResponse(
                {"detail": "Gmail connection expired, reauthorize", "code": "gmail-reauthorize"},
                status_code=503,
            )
        raise HTTPException(503, "Gmail authorization could not be refreshed. Reauthorize if the error continues.") from None
    except TransportError:
        logger.warning("Inbox check failed: Google transport unavailable")
        raise HTTPException(503, "Cannot reach Google. Check the server network, DNS, and proxy configuration.") from None
    except HttpError as exc:
        logger.warning("Inbox check failed: Gmail HTTP %s", exc.resp.status)
        raise HTTPException(503, "Gmail rejected the request. Check account permissions and Gmail API access.") from None
    except Exception as exc:
        logger.warning("Inbox check failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Inbox check unavailable. Check Gmail and worker configuration.") from None
    finally:
        poll_lock.release()


@app.post("/api/queue/{event_id}/send", dependencies=[Depends(require_dashboard_request)])
def send(event_id: int, username: Reviewer) -> dict:
    if not settings.delivery_enabled:
        raise HTTPException(409, "Delivery is disabled")
    from app.delivery import send_approved

    try:
        return send_approved(event_id, username)
    except ValueError:
        raise HTTPException(409, "Approved draft unavailable for delivery. Check the review and delivery status.") from None
    except Exception as exc:
        logger.error("Delivery failed (%s)", type(exc).__name__)
        raise HTTPException(503, "Delivery could not be confirmed. Review Gmail before another attempt.") from None


@app.get("/api/knowledge")
def knowledge(_: Reviewer) -> list[dict]:
    from app import db

    with db.connect() as conn:
        return conn.execute("select id, section, title, content, updated_at from knowledge_base order by section,id").fetchall()


@app.get("/ready")
def ready(_: Reviewer) -> dict:
    from app import db

    required = (
        settings.google_client_id, settings.google_client_secret, settings.google_refresh_token,
        settings.database_url, os.getenv("GROQ_API_KEY", ""),
    )
    if not settings.dry_run or not all(value.strip() for value in required):
        raise HTTPException(503, "Required demo configuration is missing or disabled")
    with db.connect() as conn:
        conn.execute(
            "select e.available_at,r.action,d.status,k.content from events e "
            "left join draft_reviews r on r.event_id=e.id "
            "left join deliveries d on d.review_id=r.id "
            "cross join knowledge_base k limit 0"
        )
        conn.execute("select report from evaluation_runs limit 0")
        conn.execute("select reviewer from knowledge_edits limit 0")
        count = conn.execute("select count(distinct section) as count from knowledge_base").fetchone()["count"]
    if count < 7:
        raise HTTPException(503, "Knowledge base needs seeding")
    return {"status": "ready", "delivery_enabled": settings.delivery_enabled}


class KnowledgeEdit(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    previous_content: str = Field(max_length=20000)


@app.put("/api/knowledge/{article_id}", dependencies=[Depends(require_dashboard_request)])
def edit_knowledge(article_id: int, payload: KnowledgeEdit, username: Reviewer) -> dict:
    from app import db

    content = payload.content.strip()
    if not content:
        raise HTTPException(422, "Article content is required")
    with db.connect() as conn, conn.transaction():
        article = conn.execute(
            "select content from knowledge_base where id=%s for update", (article_id,),
        ).fetchone()
        if article is None:
            raise HTTPException(404, "Article not found")
        if article["content"] != payload.previous_content:
            raise HTTPException(409, "Article changed. Reload before editing.")
        conn.execute(
            "insert into knowledge_edits (article_id,reviewer,previous_content,new_content) values (%s,%s,%s,%s)",
            (article_id, username, article["content"], content),
        )
        conn.execute(
            "update knowledge_base set content=%s, updated_at=now() where id=%s", (content, article_id),
        )
    return {"status": "updated"}


@app.get("/api/evaluations")
def evaluations(_: Reviewer) -> list[dict]:
    from app import db

    with db.connect() as conn:
        return conn.execute(
            "select id, created_at, report from evaluation_runs order by id desc limit 10"
        ).fetchall()
