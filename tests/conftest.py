import os

import pytest


@pytest.fixture(autouse=True)
def support_settings(monkeypatch):
    for name in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN", "DATABASE_URL"):
        monkeypatch.setenv(name, "test")
    monkeypatch.setenv("CLASSIFIER_MODEL", "groq:openai/gpt-oss-20b")
    monkeypatch.setenv("AGENT_MODEL", "groq:openai/gpt-oss-120b")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("DELIVERY_ENABLED", "false")
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")
    if os.getenv("PINGO_LIVE_GROQ") != "1":
        monkeypatch.setenv("GROQ_API_KEY", "test")
