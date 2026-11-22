create table evaluation_runs (
  id bigint generated always as identity primary key,
  report jsonb not null,
  created_at timestamptz not null default now()
);
alter table evaluation_runs enable row level security;

create table knowledge_edits (
  id bigint generated always as identity primary key,
  article_id bigint not null references knowledge_base(id),
  reviewer text not null,
  previous_content text not null,
  new_content text not null,
  created_at timestamptz not null default now()
);
alter table knowledge_edits enable row level security;
