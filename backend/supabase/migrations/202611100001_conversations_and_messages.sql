create table if not exists public.conversations (
    conversation_id uuid primary key default gen_random_uuid(),
    tenant_id text not null,
    actor_id text not null,
    title text,
    summary text,
    status text not null default 'active' check (
        status in ('active', 'archived', 'deleted')
    ),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists conversations_tenant_updated_idx
    on public.conversations (tenant_id, updated_at desc);

create table if not exists public.messages (
    message_id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references public.conversations(conversation_id) on delete cascade,
    tenant_id text not null,
    role text not null check (
        role in ('user', 'assistant', 'system')
    ),
    content text not null,
    created_at timestamptz not null default now()
);

create index if not exists messages_conversation_created_idx
    on public.messages (conversation_id, created_at asc);

create index if not exists messages_tenant_conversation_created_idx
    on public.messages (tenant_id, conversation_id, created_at asc);

create table if not exists public.turn_runs (
    run_id uuid primary key default gen_random_uuid(),
    conversation_id uuid not null references public.conversations(conversation_id) on delete cascade,
    user_message_id uuid references public.messages(message_id) on delete set null,
    assistant_message_id uuid references public.messages(message_id) on delete set null,
    status text not null default 'completed' check (
        status in ('queued', 'running', 'completed', 'failed')
    ),
    model text,
    route text,
    error_code text,
    latency_ms integer check (latency_ms is null or latency_ms >= 0),
    created_at timestamptz not null default now()
);

create index if not exists turn_runs_conversation_created_idx
    on public.turn_runs (conversation_id, created_at asc);

create table if not exists public.citations (
    citation_id uuid primary key default gen_random_uuid(),
    assistant_message_id uuid not null references public.messages(message_id) on delete cascade,
    citation_label text not null,
    document_id text,
    chunk_id text not null,
    page_number integer check (page_number is null or page_number >= 0),
    source text not null,
    title text,
    created_at timestamptz not null default now()
);

create index if not exists citations_assistant_message_idx
    on public.citations (assistant_message_id);

alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.turn_runs enable row level security;
alter table public.citations enable row level security;

revoke all on public.conversations from anon, authenticated;
revoke all on public.messages from anon, authenticated;
revoke all on public.turn_runs from anon, authenticated;
revoke all on public.citations from anon, authenticated;

grant select, insert, update, delete on public.conversations to service_role;
grant select, insert, update, delete on public.messages to service_role;
grant select, insert, update, delete on public.turn_runs to service_role;
grant select, insert, update, delete on public.citations to service_role;

