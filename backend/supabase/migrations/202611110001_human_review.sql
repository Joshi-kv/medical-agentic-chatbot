create unique index if not exists turn_runs_id_tenant_idx
    on public.turn_runs (run_id, tenant_id);

create table if not exists public.human_review_cases (
    case_id uuid primary key,
    tenant_id text not null,
    conversation_id uuid not null,
    run_id uuid not null,

    review_type text not null check (
        review_type in ('clinical', 'scheduling')
    ),

    status text not null default 'pending' check (
        status in ('pending', 'claimed', 'resolved', 'rejected', 'expired')
    ),

    requested_action text not null,
    context jsonb not null default '{}'::jsonb,

    claimed_by text,
    claimed_at timestamptz,

    decision text check (
        decision is null or decision in ('approved', 'rejected')
    ),
    resolution jsonb,
    resolved_by text,
    resolved_at timestamptz,

    expires_at timestamptz not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    unique (tenant_id, run_id),

    foreign key (conversation_id, tenant_id)
        references public.conversations (conversation_id, tenant_id)
        on delete cascade,

    foreign key (run_id, tenant_id)
        references public.turn_runs (run_id, tenant_id)
        on delete cascade
);

create index if not exists human_review_pending_idx
    on public.human_review_cases (
        tenant_id,
        status,
        created_at
    )
    where status in ('pending', 'claimed');

alter table public.human_review_cases enable row level security;

revoke all on public.human_review_cases from anon, authenticated;

grant select, insert, update, delete
    on public.human_review_cases
    to service_role;


create or replace function public.create_human_review_case(
    p_case_id uuid,
    p_tenant_id text,
    p_conversation_id uuid,
    p_run_id uuid,
    p_review_type text,
    p_requested_action text,
    p_context jsonb,
    p_expires_at timestamptz,
    p_created_at timestamptz
)
returns setof public.human_review_cases
language plpgsql
security definer
set search_path = public
as $$
begin
    if not exists (
        select 1
        from public.turn_runs
        where turn_runs.run_id = p_run_id
          and turn_runs.tenant_id = p_tenant_id
          and turn_runs.conversation_id = p_conversation_id
    ) then
        raise exception 'turn_not_found';
    end if;

    insert into public.human_review_cases (
        case_id,
        tenant_id,
        conversation_id,
        run_id,
        review_type,
        status,
        requested_action,
        context,
        expires_at,
        created_at,
        updated_at
    ) values (
        p_case_id,
        p_tenant_id,
        p_conversation_id,
        p_run_id,
        p_review_type,
        'pending',
        p_requested_action,
        coalesce(p_context, '{}'::jsonb),
        p_expires_at,
        p_created_at,
        p_created_at
    )
    on conflict (tenant_id, run_id) do nothing;

    return query
    select review_case.*
    from public.human_review_cases as review_case
    where review_case.tenant_id = p_tenant_id
      and review_case.run_id = p_run_id;
end;
$$;


create or replace function public.claim_human_review_case(
    p_case_id uuid,
    p_tenant_id text,
    p_staff_id text
)
returns setof public.human_review_cases
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.human_review_cases
    set status = 'expired',
        updated_at = now()
    where case_id = p_case_id
      and tenant_id = p_tenant_id
      and status = 'pending'
      and expires_at <= now();

    return query
    update public.human_review_cases
    set status = 'claimed',
        claimed_by = p_staff_id,
        claimed_at = now(),
        updated_at = now()
    where case_id = p_case_id
      and tenant_id = p_tenant_id
      and status = 'pending'
      and expires_at > now()
    returning *;

    if not found then
        return query
        select review_case.*
        from public.human_review_cases as review_case
        where review_case.case_id = p_case_id
          and review_case.tenant_id = p_tenant_id
          and review_case.status = 'claimed'
          and review_case.claimed_by = p_staff_id;
    end if;
end;
$$;


create or replace function public.resolve_human_review_case(
    p_case_id uuid,
    p_tenant_id text,
    p_staff_id text,
    p_decision text,
    p_resolution jsonb
)
returns setof public.human_review_cases
language plpgsql
security definer
set search_path = public
as $$
declare
    next_status text;
begin
    if p_decision not in ('approved', 'rejected') then
        raise exception 'invalid_review_decision';
    end if;

    next_status := case
        when p_decision = 'approved' then 'resolved'
        else 'rejected'
    end;

    return query
    update public.human_review_cases
    set status = next_status,
        decision = p_decision,
        resolution = coalesce(p_resolution, '{}'::jsonb),
        resolved_by = p_staff_id,
        resolved_at = now(),
        updated_at = now()
    where case_id = p_case_id
      and tenant_id = p_tenant_id
      and status = 'claimed'
      and claimed_by = p_staff_id
    returning *;
end;
$$;


revoke all on function public.create_human_review_case(
    uuid, text, uuid, uuid, text, text, jsonb, timestamptz, timestamptz
) from public, anon, authenticated;

revoke all on function public.claim_human_review_case(
    uuid, text, text
) from public, anon, authenticated;

revoke all on function public.resolve_human_review_case(
    uuid, text, text, text, jsonb
) from public, anon, authenticated;

grant execute on function public.create_human_review_case(
    uuid, text, uuid, uuid, text, text, jsonb, timestamptz, timestamptz
) to service_role;

grant execute on function public.claim_human_review_case(
    uuid, text, text
) to service_role;

grant execute on function public.resolve_human_review_case(
    uuid, text, text, text, jsonb
) to service_role;

