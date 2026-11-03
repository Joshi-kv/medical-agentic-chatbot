create table if not exists public.documents (
    document_id uuid primary key,
    tenant_id text not null,
    idempotency_key text not null,
    filename text not null,
    title text,
    content_type text not null check (content_type = 'application/pdf'),
    size_bytes bigint not null check (size_bytes > 0),
    checksum text not null check (length(checksum) = 64),
    storage_key text not null unique,
    status text not null default 'queued' check (
        status in ('queued', 'processing', 'completed', 'failed', 'deleting', 'deleted')
    ),
    pages_processed integer not null default 0 check (pages_processed >= 0),
    pages_requiring_ocr integer not null default 0 check (pages_requiring_ocr >= 0),
    chunks_created integer not null default 0 check (chunks_created >= 0),
    vectors_upserted integer not null default 0 check (vectors_upserted >= 0),
    embedding_model text not null,
    embedding_dimension integer not null check (embedding_dimension > 0),
    pipeline_version text not null,
    error_code text,
    error_message text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (tenant_id, idempotency_key)
);

create index if not exists documents_tenant_status_idx
    on public.documents (tenant_id, status, created_at desc);

create table if not exists public.ingestion_jobs (
    job_id uuid primary key default gen_random_uuid(),
    document_id uuid not null unique references public.documents(document_id),
    tenant_id text not null,
    status text not null default 'queued' check (
        status in ('queued', 'processing', 'completed', 'failed')
    ),
    attempts integer not null default 0 check (attempts >= 0),
    max_attempts integer not null default 5 check (max_attempts > 0),
    available_at timestamptz not null default now(),
    locked_at timestamptz,
    locked_by text,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists ingestion_jobs_available_idx
    on public.ingestion_jobs (status, available_at)
    where status = 'queued';

create table if not exists public.outbox_events (
    event_id uuid primary key default gen_random_uuid(),
    event_type text not null,
    aggregate_id uuid not null,
    tenant_id text not null,
    payload jsonb not null,
    attempts integer not null default 0 check (attempts >= 0),
    available_at timestamptz not null default now(),
    published_at timestamptz,
    created_at timestamptz not null default now()
);

create index if not exists outbox_events_unpublished_idx
    on public.outbox_events (available_at, created_at)
    where published_at is null;

create table if not exists public.document_audit_events (
    audit_id uuid primary key default gen_random_uuid(),
    document_id uuid not null,
    tenant_id text not null,
    actor_id text not null,
    action text not null,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

alter table public.documents enable row level security;
alter table public.ingestion_jobs enable row level security;
alter table public.outbox_events enable row level security;
alter table public.document_audit_events enable row level security;

revoke all on public.documents from anon, authenticated;
revoke all on public.ingestion_jobs from anon, authenticated;
revoke all on public.outbox_events from anon, authenticated;
revoke all on public.document_audit_events from anon, authenticated;

grant select, insert, update, delete on public.documents to service_role;
grant select, insert, update, delete on public.ingestion_jobs to service_role;
grant select, insert, update, delete on public.outbox_events to service_role;
grant select, insert on public.document_audit_events to service_role;

create or replace function public.create_document_ingestion(
    p_document_id uuid,
    p_tenant_id text,
    p_filename text,
    p_title text,
    p_content_type text,
    p_size_bytes bigint,
    p_checksum text,
    p_storage_key text,
    p_idempotency_key text,
    p_embedding_model text,
    p_embedding_dimension integer,
    p_pipeline_version text,
    p_actor_id text,
    p_created_at timestamptz
)
returns table (document_id uuid, status text, created boolean)
language plpgsql
security definer
set search_path = public
as $$
declare
    existing_document public.documents%rowtype;
begin
    select d.*
    into existing_document
    from public.documents d
    where d.tenant_id = p_tenant_id
      and d.idempotency_key = p_idempotency_key
    for update;

    if found then
        if existing_document.checksum <> p_checksum then
            raise exception using
                errcode = 'P0001',
                message = 'idempotency_conflict';
        end if;

        return query
        select existing_document.document_id, existing_document.status, false;
        return;
    end if;

    insert into public.documents (
        document_id,
        tenant_id,
        idempotency_key,
        filename,
        title,
        content_type,
        size_bytes,
        checksum,
        storage_key,
        status,
        embedding_model,
        embedding_dimension,
        pipeline_version,
        created_at,
        updated_at
    ) values (
        p_document_id,
        p_tenant_id,
        p_idempotency_key,
        p_filename,
        p_title,
        p_content_type,
        p_size_bytes,
        p_checksum,
        p_storage_key,
        'queued',
        p_embedding_model,
        p_embedding_dimension,
        p_pipeline_version,
        p_created_at,
        p_created_at
    );

    insert into public.ingestion_jobs (document_id, tenant_id)
    values (p_document_id, p_tenant_id);

    insert into public.outbox_events (
        event_type,
        aggregate_id,
        tenant_id,
        payload
    ) values (
        'document.ingestion_requested',
        p_document_id,
        p_tenant_id,
        jsonb_build_object(
            'document_id', p_document_id,
            'tenant_id', p_tenant_id,
            'storage_key', p_storage_key,
            'pipeline_version', p_pipeline_version
        )
    );

    insert into public.document_audit_events (
        document_id,
        tenant_id,
        actor_id,
        action,
        metadata
    ) values (
        p_document_id,
        p_tenant_id,
        p_actor_id,
        'document.uploaded',
        jsonb_build_object('checksum', p_checksum, 'size_bytes', p_size_bytes)
    );

    return query select p_document_id, 'queued'::text, true;
exception
    when unique_violation then
        select d.*
        into existing_document
        from public.documents d
        where d.tenant_id = p_tenant_id
          and d.idempotency_key = p_idempotency_key;

        if found and existing_document.checksum = p_checksum then
            return query
            select existing_document.document_id, existing_document.status, false;
            return;
        end if;

        raise exception using
            errcode = 'P0001',
            message = 'idempotency_conflict';
end;
$$;

create or replace function public.request_document_deletion(
    p_document_id uuid,
    p_tenant_id text,
    p_actor_id text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
    current_status text;
begin
    select d.status
    into current_status
    from public.documents d
    where d.document_id = p_document_id
      and d.tenant_id = p_tenant_id
    for update;

    if not found then
        return false;
    end if;

    if current_status in ('deleting', 'deleted') then
        return true;
    end if;

    update public.documents
    set status = 'deleting', updated_at = now()
    where documents.document_id = p_document_id
      and documents.tenant_id = p_tenant_id;

    insert into public.outbox_events (
        event_type,
        aggregate_id,
        tenant_id,
        payload
    ) values (
        'document.deletion_requested',
        p_document_id,
        p_tenant_id,
        jsonb_build_object('document_id', p_document_id, 'tenant_id', p_tenant_id)
    );

    insert into public.document_audit_events (
        document_id,
        tenant_id,
        actor_id,
        action
    ) values (
        p_document_id,
        p_tenant_id,
        p_actor_id,
        'document.deletion_requested'
    );

    return true;
end;
$$;

revoke all on function public.create_document_ingestion(
    uuid, text, text, text, text, bigint, text, text, text,
    text, integer, text, text, timestamptz
) from public, anon, authenticated;
grant execute on function public.create_document_ingestion(
    uuid, text, text, text, text, bigint, text, text, text,
    text, integer, text, text, timestamptz
) to service_role;

revoke all on function public.request_document_deletion(uuid, text, text)
    from public, anon, authenticated;
grant execute on function public.request_document_deletion(uuid, text, text)
    to service_role;
