from unittest.mock import Mock

import pytest

from app.config import Settings
from scripts.get_refresh_token import hosted_code

REDIRECT = "https://demo.invalid/oauth/gmail/callback"


def test_gmail_credentials_use_environment_refresh_token(monkeypatch):
    from app import gmail

    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "private-fixture-token")
    monkeypatch.setattr(gmail, "settings", Settings())
    build = Mock()
    users = build.return_value.users.return_value
    users.getProfile.return_value.execute.return_value = {"emailAddress": "demo@example.invalid"}
    users.labels.return_value.list.return_value.execute.return_value = {"labels": [
        {"name": "ai-processed", "id": "id1"}, {"name": "needs-human", "id": "id2"},
    ]}
    monkeypatch.setattr(gmail, "build", build)
    gmail.Gmail()
    assert build.call_args.kwargs["credentials"].refresh_token == "private-fixture-token"


def test_hosted_oauth_requires_exact_redirect_and_state():
    assert hosted_code(REDIRECT + "?state=expected&code=fixture", REDIRECT, "expected") == "fixture"
    with pytest.raises(ValueError, match="does not match"):
        hosted_code(REDIRECT.replace("demo.invalid", "other.invalid") + "?code=fixture", REDIRECT, "expected")
    with pytest.raises(ValueError, match="state"):
        hosted_code(REDIRECT + "?state=other&code=fixture", REDIRECT, "expected")
