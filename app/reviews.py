from app import db


def list_queue() -> list[dict]:
    with db.connect() as conn:
        return conn.execute(
            """
            select e.id, e.status, e.attempts, e.created_at, t.subject,
                   e.result->>'action' as action,
                   e.result->>'section' as section,
                   e.result->>'reason' as reason,
                   r.action as review
            from events e
            left join messages m on m.gmail_message_id = e.gmail_message_id
            left join tickets t on t.id = m.ticket_id
            left join draft_reviews r on r.event_id = e.id
            where e.status <> 'skipped'
            order by e.created_at desc, e.id desc limit 100
            """
        ).fetchall()


def get_case(event_id: int) -> dict | None:
    with db.connect() as conn:
        return conn.execute(
            """
            select e.id, e.status, e.attempts, e.error, t.subject, m.body,
                   e.result->>'action' as action, e.result->>'reason' as reason,
                   e.result->>'prompt_version' as prompt_version, e.result->'stages' as stages, e.result->'sources' as sources, e.result->>'trace_id' as trace_id,
                   d.status as delivery_status,
                   coalesce(r.draft, e.result->>'reply') as draft,
                   r.action as review, r.reviewer, r.reviewed_at
            from events e
            left join messages m on m.gmail_message_id = e.gmail_message_id
            left join tickets t on t.id = m.ticket_id
            left join draft_reviews r on r.event_id = e.id
            left join deliveries d on d.review_id = r.id
            where e.id = %s and e.status <> 'skipped'
            """,
            (event_id,),
        ).fetchone()


def save_review(event_id: int, reviewer: str, action: str, draft: str) -> bool:
    with db.connect() as conn:
        return conn.execute(
            """
            insert into draft_reviews (event_id, reviewer, action, draft)
            select id, %s, %s, %s from events
            where id = %s and status = 'processed' and result->>'action' = 'reply'
            on conflict (event_id) do nothing
            returning id
            """,
            (reviewer, action, draft, event_id),
        ).fetchone() is not None
