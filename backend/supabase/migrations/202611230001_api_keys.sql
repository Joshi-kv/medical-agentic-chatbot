create table if not exists public.api_keys (
    key_id uuid primary key default gen_random_uuid(),
    tenant_id text not null,
    name text not null check (char_length(name) between 2 and 100),
    key_prefix text not null unique,
    key_hash text not null unique check (char_length(key_hash) = 64),
    scopes text[] not null default array['conversations:read', 'conversations:write'],
    created_at timestamptz not null default now(),
    revoked_at timestamptz
);

create index if not exists api_keys_active_hash_idx on public.api_keys (key_hash)
    where revoked_at is null;

alter table public.api_keys enable row level security;
revoke all on public.api_keys from anon, authenticated;
grant select, insert, update on public.api_keys to service_role;
