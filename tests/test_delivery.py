from unittest.mock import Mock, MagicMock
import base64
import email

import pytest


def test_disabled_delivery_does_not_connect(monkeypatch):
    from app import delivery

    gmail = Mock()
    monkeypatch.setattr(delivery, "Gmail", gmail)
    with pytest.raises(ValueError, match="disabled"):
        delivery.send_approved(1, "reviewer")
    gmail.assert_not_called()


def test_approved_delivery_uses_stored_text_and_never_retries_uncertain(monkeypatch):
    from app import delivery

    monkeypatch.setattr(delivery.settings, "delivery_enabled", True)
    entry = {
        "delivery_id": 7, "gmail_message_id": "original", "ticket_id": 2,
        "customer_email": "customer@example.test", "draft": "Approved exact text",
    }
    claim = Mock(side_effect=[entry, None])
    monkeypatch.setattr(delivery, "claim", claim)
    gmail = Mock(address="support@example.test")
    gmail.get_message.return_value = {"threadId": "thread", "payload": {"headers": [
        {"name": "From", "value": "customer@example.test"},
        {"name": "Message-ID", "value": "<original@example.test>"},
        {"name": "Subject", "value": "Bluetooth"},
    ]}}
    send = gmail.users.messages.return_value.send
    send.return_value.execute.side_effect = TimeoutError
    monkeypatch.setattr(delivery, "Gmail", lambda: gmail)
    conn = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = conn
    monkeypatch.setattr(delivery.db, "connect", lambda: context)

    with pytest.raises(TimeoutError):
        delivery.send_approved(1, "reviewer")
    raw = send.call_args.kwargs["body"]["raw"]
    sent = email.message_from_bytes(base64.urlsafe_b64decode(raw))
    assert sent["To"] == "customer@example.test"
    assert "Approved exact text" in sent.get_payload()
    assert "uncertain" in conn.execute.call_args.args[0]
    with pytest.raises(ValueError):
        delivery.send_approved(1, "reviewer")
    send.assert_called_once()


def test_delivery_claim_requires_approved_latest_draft(monkeypatch):
    from app import delivery

    context = MagicMock()
    conn = context.__enter__.return_value
    conn.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(delivery.db, "connect", lambda: context)
    assert delivery.claim(42, "reviewer") is None
    sql, args = conn.execute.call_args.args
    assert "r.action = 'approve'" in sql
    assert "newer.id > m.id" in sql
    assert "on conflict (review_id) do nothing" in sql
    assert args == (42, "reviewer")


def test_successful_delivery_persists_outbound_message_and_status(monkeypatch):
    from app import delivery

    monkeypatch.setattr(delivery.settings, "delivery_enabled", True)
    monkeypatch.setattr(delivery, "claim", lambda *_: {
        "delivery_id": 1, "gmail_message_id": "inbound", "ticket_id": 2,
        "customer_email": "customer@example.test", "draft": "Approved reply",
    })
    gmail = Mock(address="support@example.test")
    gmail.get_message.return_value = {"threadId": "thread", "payload": {"headers": [
        {"name": "From", "value": "customer@example.test"},
        {"name": "Message-ID", "value": "<original@example.test>"},
    ]}}
    gmail.users.messages.return_value.send.return_value.execute.return_value = {"id": "outbound"}
    monkeypatch.setattr(delivery, "Gmail", lambda: gmail)
    context = MagicMock()
    conn = context.__enter__.return_value
    monkeypatch.setattr(delivery.db, "connect", lambda: context)
    assert delivery.send_approved(3, "reviewer") == {"status": "sent", "gmail_message_id": "outbound"}
    assert conn.execute.call_args_list[0].args[1] == (2, "outbound", "support@example.test", "Approved reply")
    assert "status='answered'" in conn.execute.call_args_list[1].args[0]
    assert "status='sent'" in conn.execute.call_args_list[2].args[0]
