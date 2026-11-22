import pytest

from app.ai.grounding import CitationError, verify_sources
from app.ai.schemas import AgentReply, ClaimSource


def draft(claim="Enable Bluetooth.", snippet="Enable Bluetooth.", article_id=7):
    return AgentReply(
        action="reply", reply=claim, reasoning="Known guidance",
        enough_information=True, promises_an_action=False,
        sources=[ClaimSource(claim=claim, article_id=article_id, snippet=snippet)],
    )


ARTICLES = [{"id": 7, "content": "Enable Bluetooth. Charge the earbuds."}]


def test_verified_quote_allows_whitespace_variations():
    verify_sources(draft(snippet="Enable\nBluetooth."), ARTICLES)


@pytest.mark.parametrize("answer", [
    draft(article_id=999),
    draft(claim="The case LED flashes.", snippet="The case LED flashes."),
    draft(snippet="Customer says the LED flashes."),
    draft(snippet=" "),
])
def test_invalid_source_is_rejected(answer):
    with pytest.raises(CitationError):
        verify_sources(answer, ARTICLES)


def test_missing_citations_are_rejected():
    answer = draft()
    answer.sources = []
    with pytest.raises(CitationError):
        verify_sources(answer, ARTICLES)


def test_claim_must_appear_in_draft():
    answer = draft()
    answer.reply = "Different text"
    with pytest.raises(CitationError):
        verify_sources(answer, ARTICLES)


def test_quote_integrity_does_not_establish_entailment():
    # A real quote can still be misused. The semantic grader must catch this.
    verify_sources(draft(claim="The case LED flashes.", snippet="Enable Bluetooth."), ARTICLES)
