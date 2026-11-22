"""Explicit approved delivery. An uncertain send requires manual reconciliation."""
from email.message import EmailMessage
from email.utils import parseaddr
import base64

from app import db
from app.config import settings
from app.gmail import Gmail, get_headers


def claim(event_id: int, reviewer: str) -> dict | None:
    with db.connect() as conn:
        return conn.execute(
            """
            with candidate as (
              select r.id as review_id, r.draft, e.gmail_message_id,
                     m.ticket_id, t.customer_email
              from draft_reviews r
              join events e on e.id = r.event_id
              join messages m on m.gmail_message_id = e.gmail_message_id
              join tickets t on t.id = m.ticket_id
              where e.id = %s and r.action = 'approve'
                and e.status = 'processed' and e.result->>'action' = 'reply'
                and t.status <> 'escalated'
                and not exists (
                  select 1 from messages newer
                  where newer.ticket_id = t.id and newer.direction = 'in'
                    and newer.id > m.id
                )
            ), claimed as (
              insert into deliveries (review_id, requested_by)
              select review_id, %s from candidate
              on conflict (review_id) do nothing returning id, review_id
            )
            select c.*, claimed.id as delivery_id
            from candidate c join claimed using (review_id)
            """,
            (event_id, reviewer),
        ).fetchone()


def send_approved(event_id: int, reviewer: str) -> dict:
    if not settings.delivery_enabled:
        raise ValueError("Delivery is disabled")
    gmail = Gmail()
    entry = claim(event_id, reviewer)
    if entry is None:
        raise ValueError("Approved draft unavailable, superseded, or delivery already attempted")
    try:
        original = gmail.get_message(entry["gmail_message_id"])
        headers = get_headers(original)
        recipient = parseaddr(headers.get("from", ""))[1]
        if not recipient or recipient.casefold() != entry["customer_email"].casefold():
            raise ValueError("Recipient differs from ticket")
        if not headers.get("message-id"):
            raise ValueError("Original email lacks Message-ID")
        reply = EmailMessage()
        reply["To"] = recipient
        reply["From"] = gmail.address
        subject = headers.get("subject", "")
        reply["Subject"] = subject if subject.lower().startswith("re:") else "Re: " + subject
        reply["In-Reply-To"] = headers["message-id"]
        reply["References"] = headers["message-id"]
        reply.set_content(entry["draft"])
        sent = gmail.users.messages().send(
            userId="me", body={
                "raw": base64.urlsafe_b64encode(reply.as_bytes()).decode(),
                "threadId": original["threadId"],
            },
        ).execute()
        with db.connect() as conn, conn.transaction():
            conn.execute(
                """insert into messages (ticket_id,gmail_message_id,direction,from_address,body)
                   values (%s,%s,'out',%s,%s)""",
                (entry["ticket_id"], sent["id"], gmail.address, entry["draft"]),
            )
            conn.execute("update tickets set status='answered', updated_at=now() where id=%s", (entry["ticket_id"],))
            conn.execute(
                "update deliveries set status='sent', gmail_message_id=%s, updated_at=now() where id=%s",
                (sent["id"], entry["delivery_id"]),
            )
        return {"status": "sent", "gmail_message_id": sent["id"]}
    except Exception:
        with db.connect() as conn:
            conn.execute(
                "update deliveries set status='uncertain', updated_at=now() where id=%s",
                (entry["delivery_id"],),
            )
        raise
