create table deliveries (
  id bigint generated always as identity primary key,
  review_id bigint not null unique references draft_reviews(id),
  status text not null default 'sending' check (status in ('sending', 'sent', 'uncertain')),
  requested_by text not null,
  gmail_message_id text unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
alter table deliveries enable row level security;
