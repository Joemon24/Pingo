from collections import deque
import importlib
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def dashboard(monkeypatch):
    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "failed_logins", deque())
    monkeypatch.setattr(main.settings, "dashboard_username", "reviewer")
    monkeypatch.setattr(main.settings, "dashboard_password", "test-password")
    return main, TestClient(main.app)


def test_dashboard_and_queue_require_authentication(dashboard, monkeypatch):
    main, client = dashboard
    read = Mock(return_value=[])
    monkeypatch.setattr(main.reviews, "list_queue", read)
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 401
    assert client.get("/api/queue").status_code == 401
    read.assert_not_called()
    auth = ("reviewer", "test-password")
    assert "Support inbox" in client.get("/", auth=auth).text
    assert client.get("/api/queue", auth=auth).json() == []


def test_review_requires_header_and_saves_once(dashboard, monkeypatch):
    main, client = dashboard
    save = Mock(side_effect=[True, False])
    monkeypatch.setattr(main.reviews, "save_review", save)
    url = "/api/queue/42/review"
    auth = ("reviewer", "test-password")
    headers = {"X-Pingo-Action": "dashboard"}
    data = {"action": "approve", "draft": " Check Bluetooth. "}
    assert client.post(url, auth=auth, json=data).status_code == 403
    save.assert_not_called()
    assert client.post(url, auth=auth, headers=headers, json=data).json() == {"review": "approve"}
    save.assert_called_once_with(42, "reviewer", "approve", "Check Bluetooth.")
    assert client.post(url, auth=auth, headers=headers, json=data).status_code == 409


def test_poll_calls_worker_with_bounded_batch(dashboard, monkeypatch):
    _, client = dashboard
    from app import poller

    worker = Mock(return_value={"found": 1, "failed": 0})
    monkeypatch.setattr(poller, "poll_once", worker)
    response = client.post(
        "/api/poll", auth=("reviewer", "test-password"),
        headers={"X-Pingo-Action": "dashboard"},
    )
    assert response.json() == {"found": 1, "failed": 0}
    worker.assert_called_once_with(max_results=1)


def test_delivery_requires_authentication_and_explicit_dashboard_request(dashboard, monkeypatch):
    main, client = dashboard
    monkeypatch.setattr(main.settings, "delivery_enabled", True)
    from app import delivery

    send = Mock(return_value={"status": "sent"})
    monkeypatch.setattr(delivery, "send_approved", send)
    url = "/api/queue/42/send"
    assert client.post(url, headers={"X-Pingo-Action": "dashboard"}).status_code == 401
    assert client.post(url, auth=("reviewer", "test-password")).status_code == 403
    send.assert_not_called()
    result = client.post(url, auth=("reviewer", "test-password"), headers={"X-Pingo-Action": "dashboard"})
    assert result.json() == {"status": "sent"}
    send.assert_called_once_with(42, "reviewer")


def test_knowledge_edit_conflict_preserves_article(dashboard, monkeypatch):
    from unittest.mock import MagicMock
    from app import db

    _, client = dashboard
    context = MagicMock()
    conn = context.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = {"content": "Newer policy"}
    monkeypatch.setattr(db, "connect", lambda: context)
    response = client.put(
        "/api/knowledge/1", auth=("reviewer", "test-password"),
        headers={"X-Pingo-Action": "dashboard"},
        json={"previous_content": "Old policy", "content": "My edit"},
    )
    assert response.status_code == 409
    assert conn.execute.call_count == 1


def test_evaluation_results_require_login(dashboard, monkeypatch):
    from app import db

    _, client = dashboard
    connect = Mock()
    monkeypatch.setattr(db, "connect", connect)
    assert client.get("/api/evaluations").status_code == 401
    connect.assert_not_called()


@pytest.mark.parametrize(("error_type", "expected"), [
    ("refresh", "Reauthorize"),
    ("transport", "Cannot reach Google"),
])
def test_poll_reports_safe_actionable_google_errors(dashboard, monkeypatch, error_type, expected):
    from app import poller
    from google.auth.exceptions import RefreshError, TransportError

    _, client = dashboard
    error = RefreshError("private provider detail") if error_type == "refresh" else TransportError("private provider detail")
    monkeypatch.setattr(poller, "poll_once", Mock(side_effect=error))
    response = client.post(
        "/api/poll", auth=("reviewer", "test-password"),
        headers={"X-Pingo-Action": "dashboard"},
    )
    assert response.status_code == 503
    assert expected in response.json()["detail"]
    assert "private provider detail" not in response.text


def configure_ready(monkeypatch, main):
    for key in ("google_client_id", "google_client_secret", "google_refresh_token", "database_url"):
        monkeypatch.setattr(main.settings, key, "private fixture")
    monkeypatch.setattr(main.settings, "dry_run", True)
    monkeypatch.setenv("GROQ_API_KEY", "private fixture")


def test_health_is_independent_of_db_when_readiness_returns_503(dashboard, monkeypatch):
    import psycopg
    from app import db

    main, client = dashboard
    configure_ready(monkeypatch, main)
    connect = Mock(side_effect=psycopg.OperationalError("private fixture"))
    monkeypatch.setattr(db, "connect", connect)
    assert client.get("/health").json() == {"status": "ok"}
    connect.assert_not_called()
    response = client.get("/ready", auth=("reviewer", "test-password"))
    assert response.status_code == 503
    assert "private fixture" not in response.text
    connect.assert_called_once()


def test_readiness_checks_config_and_required_database_tables(dashboard, monkeypatch):
    from unittest.mock import MagicMock
    from app import db

    main, client = dashboard
    configure_ready(monkeypatch, main)
    connect = MagicMock()
    conn = connect.return_value.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = {"count": 7}
    monkeypatch.setattr(db, "connect", connect)
    auth = ("reviewer", "test-password")
    assert client.get("/ready").status_code == 401
    assert client.get("/ready", auth=auth).json()["status"] == "ready"
    assert conn.execute.call_count == 4
    connect.reset_mock()
    monkeypatch.delenv("GROQ_API_KEY")
    assert client.get("/ready", auth=auth).status_code == 503
    connect.assert_not_called()


def test_invalid_grant_has_redacted_reauthorization_status(dashboard, monkeypatch, caplog):
    from app import poller
    from google.auth.exceptions import RefreshError

    _, client = dashboard
    private = "private provider credential"
    gmail = Mock()
    gmail.list_new_message_ids.return_value = ["demo-id"]
    monkeypatch.setattr(poller, "Gmail", Mock(return_value=gmail))
    monkeypatch.setattr(poller, "handle_message", Mock(side_effect=RefreshError(
        private, {"error": "invalid_grant", "error_description": private},
    )))
    monkeypatch.setattr(poller.settings, "dry_run", True)
    response = client.post("/api/poll", auth=("reviewer", "test-password"),
                           headers={"X-Pingo-Action": "dashboard"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Gmail connection expired, reauthorize", "code": "gmail-reauthorize"}
    assert private not in response.text + caplog.text
    assert 'id="gmail-warning" role="alert"' in client.get("/", auth=("reviewer", "test-password")).text


def test_hosted_callback_does_not_echo_query(dashboard, monkeypatch):
    main, client = dashboard
    monkeypatch.setattr(main.settings, "google_redirect_uri", "https://demo.invalid/oauth/gmail/callback")
    response = client.get("/oauth/gmail/callback?code=private-code&state=private-state")
    assert response.status_code == 200
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"
    assert "private" not in response.text


def test_login_is_temporarily_locked_after_five_failures(dashboard, monkeypatch):
    main, client = dashboard
    clock = Mock(return_value=100.0)
    monkeypatch.setattr(main, "monotonic", clock)
    for _ in range(8):
        assert client.get("/").status_code == 401
    for _ in range(5):
        assert client.get("/", auth=("reviewer", "wrong")).status_code == 401
    valid = ("reviewer", "test-password")
    locked = client.get("/api/status", auth=valid)
    assert locked.status_code == 429
    assert locked.headers["Retry-After"] == "60"
    assert client.get("/health").status_code == 200
    clock.return_value += 60
    assert client.get("/", auth=valid).status_code == 200


def test_disabled_delivery_is_visible_and_never_invokes_sender(dashboard, monkeypatch):
    from app import delivery

    main, client = dashboard
    monkeypatch.setattr(main.settings, "delivery_enabled", False)
    send = Mock()
    monkeypatch.setattr(delivery, "send_approved", send)
    auth = ("reviewer", "test-password")
    assert "Delivery disabled" in client.get("/", auth=auth).text
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", auth=auth).json() == {"delivery_enabled": False}
    response = client.post("/api/queue/42/send", auth=auth, headers={"X-Pingo-Action": "dashboard"})
    assert response.status_code == 409
    assert response.json() == {"detail": "Delivery is disabled"}
    send.assert_not_called()


def test_invalid_dashboard_payload_does_not_echo_input(dashboard):
    _, client = dashboard
    response = client.post("/api/queue/42/review", auth=("reviewer", "test-password"),
                           headers={"X-Pingo-Action": "dashboard"},
                           json={"action": "private accidentally pasted value", "draft": ""})
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request data"}
