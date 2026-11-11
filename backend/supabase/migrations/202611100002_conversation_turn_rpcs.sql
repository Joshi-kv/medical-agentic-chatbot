alter table public.turn_runs
    add column if not exists tenant_id text;

update public.turn_runs as run
set tenant_id = conversation.tenant_id
from public.conversations as conversation
where run.conversation_id = conversation.conversation_id
  and run.tenant_id is null;

alter table public.turn_runs
    alter column tenant_id set not null;

alter table public.citations
    add column if not exists tenant_id text;

update public.citations as citation
set tenant_id = message.tenant_id
from public.messages as message
where citation.assistant_message_id = message.message_id
  and citation.tenant_id is null;

alter table public.citations
    alter column tenant_id set not null;

create unique index if not exists conversations_id_tenant_idx
    on public.conversations (conversation_id, tenant_id);

create unique index if not exists messages_id_tenant_idx
    on public.messages (message_id, tenant_id);

create index if not exists turn_runs_tenant_conversation_created_idx
    on public.turn_runs (tenant_id, conversation_id, created_at asc);

create index if not exists citations_tenant_message_idx
    on public.citations (tenant_id, assistant_message_id);

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'messages_conversation_tenant_fk'
    ) then
        alter table public.messages
            add constraint messages_conversation_tenant_fk
            foreign key (conversation_id, tenant_id)
            references public.conversations (conversation_id, tenant_id)
            on delete cascade;
    end if;

    if not exists (
        select 1
        from pg_constraint
        where conname = 'turn_runs_conversation_tenant_fk'
    ) then
        alter table public.turn_runs
            add constraint turn_runs_conversation_tenant_fk
            foreign key (conversation_id, tenant_id)
            references public.conversations (conversation_id, tenant_id)
            on delete cascade;
    end if;

    if not exists (
        select 1
        from pg_constraint
        where conname = 'citations_message_tenant_fk'
    ) then
        alter table public.citations
            add constraint citations_message_tenant_fk
            foreign key (assistant_message_id, tenant_id)
            references public.messages (message_id, tenant_id)
            on delete cascade;
    end if;
end
$$;

create or replace function public.start_conversation_turn(
    p_run_id uuid,
    p_user_message_id uuid,
    p_conversation_id uuid,
    p_tenant_id text,
    p_content text,
    p_created_at timestamptz
)
returns table (
    run_id uuid,
    message_id uuid,
    conversation_id uuid,
    tenant_id text,
    role text,
    content text,
    created_at timestamptz
)
language plpgsql
security definer
set search_path = public
as $$
begin
    if not exists (
        select 1
        from public.conversations
        where conversations.conversation_id = p_conversation_id
          and conversations.tenant_id = p_tenant_id
          and conversations.status = 'active'
    ) then
        raise exception 'conversation_not_found';
    end if;

    insert into public.messages (
        message_id,
        conversation_id,
        tenant_id,
        role,
        content,
        created_at
    ) values (
        p_user_message_id,
        p_conversation_id,
        p_tenant_id,
        'user',
        p_content,
        p_created_at
    );

    insert into public.turn_runs (
        run_id,
        conversation_id,
        tenant_id,
        user_message_id,
        status,
        created_at
    ) values (
        p_run_id,
        p_conversation_id,
        p_tenant_id,
        p_user_message_id,
        'running',
        p_created_at
    );

    update public.conversations
    set updated_at = p_created_at
    where conversations.conversation_id = p_conversation_id
      and conversations.tenant_id = p_tenant_id;

    return query
    select
        p_run_id,
        message.message_id,
        message.conversation_id,
        message.tenant_id,
        message.role,
        message.content,
        message.created_at
    from public.messages as message
    where message.message_id = p_user_message_id
      and message.tenant_id = p_tenant_id;
end;
$$;

create or replace function public.complete_conversation_turn(
    p_run_id uuid,
    p_tenant_id text,
    p_assistant_message_id uuid,
    p_content text,
    p_citations jsonb,
    p_model text,
    p_route text,
    p_latency_ms integer,
    p_created_at timestamptz
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    target_conversation_id uuid;
    citation jsonb;
begin
    select turn_runs.conversation_id
    into target_conversation_id
    from public.turn_runs
    where turn_runs.run_id = p_run_id
      and turn_runs.tenant_id = p_tenant_id
      and turn_runs.status = 'running'
    for update;

    if target_conversation_id is null then
        raise exception 'turn_not_found';
    end if;

    insert into public.messages (
        message_id,
        conversation_id,
        tenant_id,
        role,
        content,
        created_at
    ) values (
        p_assistant_message_id,
        target_conversation_id,
        p_tenant_id,
        'assistant',
        p_content,
        p_created_at
    );

    for citation in
        select value
        from jsonb_array_elements(coalesce(p_citations, '[]'::jsonb))
    loop
        insert into public.citations (
            citation_id,
            tenant_id,
            assistant_message_id,
            citation_label,
            document_id,
            chunk_id,
            page_number,
            source,
            title,
            created_at
        ) values (
            (citation ->> 'citation_id')::uuid,
            p_tenant_id,
            p_assistant_message_id,
            citation ->> 'citation_label',
            citation ->> 'document_id',
            citation ->> 'chunk_id',
            (citation ->> 'page_number')::integer,
            citation ->> 'source',
            citation ->> 'title',
            p_created_at
        );
    end loop;

    update public.turn_runs
    set assistant_message_id = p_assistant_message_id,
        status = 'completed',
        model = p_model,
        route = p_route,
        latency_ms = p_latency_ms,
        error_code = null
    where turn_runs.run_id = p_run_id
      and turn_runs.tenant_id = p_tenant_id;

    update public.conversations
    set updated_at = p_created_at
    where conversations.conversation_id = target_conversation_id
      and conversations.tenant_id = p_tenant_id;

    return true;
end;
$$;

create or replace function public.fail_conversation_turn(
    p_run_id uuid,
    p_tenant_id text,
    p_error_code text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.turn_runs
    set status = 'failed',
        error_code = p_error_code
    where turn_runs.run_id = p_run_id
      and turn_runs.tenant_id = p_tenant_id
      and turn_runs.status in ('queued', 'running');

    return found;
end;
$$;

revoke all on function public.start_conversation_turn(
    uuid, uuid, uuid, text, text, timestamptz
) from public, anon, authenticated;

revoke all on function public.complete_conversation_turn(
    uuid, text, uuid, text, jsonb, text, text, integer, timestamptz
) from public, anon, authenticated;

revoke all on function public.fail_conversation_turn(
    uuid, text, text
) from public, anon, authenticated;

grant execute on function public.start_conversation_turn(
    uuid, uuid, uuid, text, text, timestamptz
) to service_role;

grant execute on function public.complete_conversation_turn(
    uuid, text, uuid, text, jsonb, text, text, integer, timestamptz
) to service_role;

grant execute on function public.fail_conversation_turn(
    uuid, text, text
) to service_role;
