import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.config import settings


def connect() -> psycopg.Connection:
    if not settings.database_url.strip():
        raise psycopg.OperationalError("DATABASE_URL is required")
    return psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row, connect_timeout=5)


def start_event(gmail_message_id: str, max_attempts: int) -> dict | None:
    """Claim a new or due event. The attempt number identifies the claim."""
    with connect() as conn, conn.transaction():
        conn.execute(
            """
            update events set status = 'failed', error = 'Final processing lease expired'
            where gmail_message_id = %s and status = 'processing'
              and attempts >= %s and available_at <= now()
            """,
            (gmail_message_id, max_attempts),
        )
        return conn.execute(
            """
            insert into events (gmail_message_id, available_at)
            values (%s, now() + interval '5 minutes')
            on conflict (gmail_message_id) do update
              set attempts = events.attempts + 1, status = 'processing', error = null,
                  available_at = excluded.available_at
              where events.status in ('processing', 'failed') and events.attempts < %s
                and events.available_at <= now()
            returning id, attempts
            """,
            (gmail_message_id, max_attempts),
        ).fetchone()


def get_event(gmail_message_id: str) -> dict | None:
    with connect() as conn:
        return conn.execute(
            "select status, result, attempts from events where gmail_message_id = %s",
            (gmail_message_id,),
        ).fetchone()


def finish_event(event_id: int, attempt: int, status: str, result: dict) -> bool:
    with connect() as conn, conn.transaction():
        updated = conn.execute(
            """
            update events set status = %s, result = %s, processed_at = now()
            where id = %s and attempts = %s and status = 'processing'
            """,
            (status, Jsonb(result), event_id, attempt),
        )
        if updated.rowcount == 0:
            return False

        if status == "processed":
            conn.execute(
                """
                update tickets
                set status = %s, category = %s, proposed_action = %s, updated_at = now()
                where id = %s and status <> 'escalated'
                """,
                (
                    "escalated" if result["action"] == "escalate" else "open",
                    result["section"], result["proposed_action"], result["ticket_id"],
                ),
            )
        return True


def fail_event(event_id: int, attempt: int, error: str) -> bool:
    delay_seconds = 30 * 2 ** (attempt - 1)
    with connect() as conn:
        updated = conn.execute(
            """
            update events set status = 'failed', error = %s,
                available_at = now() + %s * interval '1 second'
            where id = %s and attempts = %s and status = 'processing'
            """,
            (error, delay_seconds, event_id, attempt),
        )
        return updated.rowcount == 1


def upsert_ticket(gmail_thread_id: str, customer_email: str, subject: str) -> int:
    with connect() as conn:
        row = conn.execute(
            """
            insert into tickets (gmail_thread_id, customer_email, subject)
            values (%s, %s, %s)
            on conflict (gmail_thread_id) do update set updated_at = now()
            returning id
            """,
            (gmail_thread_id, customer_email, subject),
        ).fetchone()
        return row["id"]


def ticket_needs_review(ticket_id: int) -> bool:
    with connect() as conn:
        row = conn.execute("select status from tickets where id = %s", (ticket_id,)).fetchone()
        return row is not None and row["status"] == "escalated"


def save_message(
    ticket_id: int, gmail_message_id: str, direction: str, from_address: str, body: str
) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into messages (ticket_id, gmail_message_id, direction, from_address, body)
            values (%s, %s, %s, %s, %s)
            on conflict (gmail_message_id) do nothing
            """,
            (ticket_id, gmail_message_id, direction, from_address, body),
        )


def get_section(section: str) -> list[dict]:
    with connect() as conn:
        return conn.execute(
            "select id, title, content from knowledge_base where section = %s order by id",
            (section,),
        ).fetchall()
