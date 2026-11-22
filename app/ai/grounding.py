"""Citation integrity checks. Semantic claim support still requires grading."""
from app.ai.schemas import AgentReply


class CitationError(ValueError):
    pass


def normalize(text: str) -> str:
    """Allow whitespace differences while preserving the actual words and case."""
    return " ".join(text.split())


def verify_sources(answer: AgentReply, articles: list[dict]) -> None:
    if answer.action != "reply":
        return
    if not answer.sources:
        raise CitationError("Reply has no evidence")
    by_id = {article["id"]: article["content"] for article in articles}
    reply = normalize(answer.reply)
    for source in answer.sources:
        content = by_id.get(source.article_id)
        if content is None:
            raise CitationError("Article was not retrieved")
        snippet = normalize(source.snippet)
        claim = normalize(source.claim)
        if not snippet or snippet not in normalize(content):
            raise CitationError("Snippet is not present in the retrieved article")
        if not claim or claim not in reply:
            raise CitationError("Cited claim is not present in the reply")
