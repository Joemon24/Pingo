from unittest.mock import MagicMock, Mock, call

import psycopg
import pytest


def test_dry_run_persists_then_recovers_labels_without_duplicate(monkeypatch):
    from app import pipeline

    gmail = Mock(address="support@example.test")
    gmail.get_message.return_value = {"threadId": "thread", "payload": {}, "snippet": "Refund please"}
    gmail.get_thread.return_value = {"messages": [gmail.get_message.return_value]}
    gmail.is_escalated.return_value = False
    gmail.add_label.side_effect = [RuntimeError, None, None]
    result = {}
    start = Mock(side_effect=[{"id": 1, "attempts": 1}, None])
    monkeypatch.setattr(pipeline.db, "start_event", start)
    monkeypatch.setattr(pipeline.db, "upsert_ticket", Mock(return_value=2))
    save = Mock()
    monkeypatch.setattr(pipeline.db, "save_message", save)
    monkeypatch.setattr(pipeline.db, "ticket_needs_review", lambda _: False)
    fail = Mock()
    monkeypatch.setattr(pipeline.db, "fail_event", fail)

    def finish(event_id, attempt, status, decision):
        assert (event_id, attempt) == (1, 1)
        result.update(status=status, result=decision)
        return True

    monkeypatch.setattr(pipeline.db, "finish_event", finish)
    monkeypatch.setattr(pipeline.db, "get_event", lambda _: result)
    with pytest.raises(RuntimeError):
        pipeline.handle_message(gmail, "message")
    pipeline.handle_message(gmail, "message")
    assert result["result"]["dry_run"] is True
    assert result["result"]["action"] == "escalate"
    assert start.call_args_list == [call("message", 3), call("message", 3)]
    assert gmail.add_label.call_args_list[-2:] == [
        call("message", pipeline.ESCALATED_LABEL), call("message", pipeline.PROCESSED_LABEL),
    ]
    gmail.get_message.assert_called_once()
    save.assert_called_once()
    fail.assert_not_called()
    gmail.send_reply.assert_not_called()


def test_database_failure_never_fetches_or_sends(monkeypatch):
    from app import pipeline

    gmail = Mock()
    monkeypatch.setattr(pipeline.db, "start_event", Mock(side_effect=psycopg.OperationalError))
    with pytest.raises(psycopg.OperationalError):
        pipeline.handle_message(gmail, "message")
    assert gmail.mock_calls == []


def test_live_delivery_mode_is_rejected(monkeypatch):
    from app import pipeline

    monkeypatch.setattr(pipeline.settings, "dry_run", False)
    gmail = Mock()
    with pytest.raises(RuntimeError, match="DRY_RUN=true"):
        pipeline.handle_message(gmail, "message")
    assert gmail.mock_calls == []


def test_event_claim_enforces_attempt_limit(monkeypatch):
    from app import db

    connection = MagicMock()
    context = Mock()
    context.__enter__ = Mock(return_value=connection)
    context.__exit__ = Mock(return_value=False)
    connection.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(db, "connect", lambda: context)
    assert db.start_event("gmail-message", 3) is None
    sql, params = connection.execute.call_args.args
    assert "events.attempts < %s" in sql
    assert "events.available_at <= now()" in sql
    assert "events.status in ('processing', 'failed')" in sql
    assert params == ("gmail-message", 3)


@pytest.mark.parametrize(("attempts", "expected_labels"), [
    (2, []),
    (3, ["needs-human", "ai-processed"]),
])
def test_failed_event_waits_or_routes_to_review(monkeypatch, attempts, expected_labels):
    from app import pipeline

    gmail = Mock()
    monkeypatch.setattr(pipeline.db, "start_event", Mock(return_value=None))
    monkeypatch.setattr(pipeline.db, "get_event", Mock(return_value={
        "status": "failed", "attempts": attempts,
    }))
    process = Mock()
    monkeypatch.setattr(pipeline, "process", process)

    pipeline.handle_message(gmail, "message")

    process.assert_not_called()
    assert gmail.add_label.call_args_list == [call("message", label) for label in expected_labels]


def test_superseded_attempt_does_not_label_message(monkeypatch):
    from app import pipeline

    gmail = Mock()
    monkeypatch.setattr(pipeline.db, "start_event", Mock(return_value={"id": 1, "attempts": 1}))
    monkeypatch.setattr(pipeline, "process", Mock(return_value=("processed", {})))
    finish = Mock(return_value=False)
    monkeypatch.setattr(pipeline.db, "finish_event", finish)

    pipeline.handle_message(gmail, "message")

    finish.assert_called_once()
    assert finish.call_args.args[:3] == (1, 1, "processed")
    assert finish.call_args.args[3]["trace_id"]
    gmail.add_label.assert_not_called()
