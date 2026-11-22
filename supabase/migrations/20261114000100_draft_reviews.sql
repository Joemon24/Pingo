create table draft_reviews (
  id bigint generated always as identity primary key,
  event_id bigint not null references events (id),
  reviewer text not null,
  action text not null check (action in ('approve', 'reject')),
  draft text not null,
  reviewed_at timestamptz not null default now(),
  unique (event_id)
);

alter table draft_reviews enable row level security;
