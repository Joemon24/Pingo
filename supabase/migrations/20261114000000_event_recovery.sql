alter table events
  add column available_at timestamptz not null default now();
