import json
from pathlib import Path

import pytest

from app.ai.grounding import CitationError, verify_sources
from app.ai.schemas import AgentReply


def test_saved_bluetooth_citations():
    fixture = json.loads(Path("tests/fixtures/bluetooth-citations.json").read_text())
    draft = AgentReply.model_validate(fixture["draft"])
    with pytest.raises(CitationError, match="claim is not present"):
        verify_sources(draft, fixture["articles"])
    draft.sources[1].claim = "Make sure Bluetooth is enabled on your phone"
    draft.sources[2].claim = "remove the earbuds from the phone's previously paired devices list and try pairing again."
    verify_sources(draft, fixture["articles"])
    draft.sources[1].claim = "The indicator flashes blue."
    with pytest.raises(CitationError, match="claim is not present"):
        verify_sources(draft, fixture["articles"])
