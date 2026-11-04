create or replace function public.claim_ingestion_job(
    p_worker_id text,
    p_lease_seconds integer
)
returns table (
    job_id uuid,
    document_id uuid,
    tenant_id text,
    filename text,
    title text,
    content_type text,
    size_bytes bigint,
    checksum text,
    storage_key text,
    attempts integer,
    max_attempts integer
)
language plpgsql
security definer
set search_path = public
as $$
declare
    selected_job public.ingestion_jobs%rowtype;
begin
    select j.*
    into selected_job
    from public.ingestion_jobs j
    where (
        (
            j.status = 'queued'
            and j.available_at <= now()
            and j.attempts < j.max_attempts
        )
        or (
            j.status = 'processing'
            and j.locked_at < now() - make_interval(
                secs => greatest(p_lease_seconds, 1)
            )
        )
    )
    order by j.available_at, j.created_at
    for update skip locked
    limit 1;

    if not found then
        return;
    end if;

    update public.ingestion_jobs j
    set status = 'processing',
        attempts = case
            when selected_job.status = 'queued' then j.attempts + 1
            else j.attempts
        end,
        locked_at = now(),
        locked_by = p_worker_id,
        updated_at = now()
    where j.job_id = selected_job.job_id
    returning j.* into selected_job;

    update public.documents d
    set status = 'processing',
        error_code = null,
        error_message = null,
        updated_at = now()
    where d.document_id = selected_job.document_id
      and d.tenant_id = selected_job.tenant_id;

    return query
    select
        selected_job.job_id,
        d.document_id,
        d.tenant_id,
        d.filename,
        d.title,
        d.content_type,
        d.size_bytes,
        d.checksum,
        d.storage_key,
        selected_job.attempts,
        selected_job.max_attempts
    from public.documents d
    where d.document_id = selected_job.document_id
      and d.tenant_id = selected_job.tenant_id;
end;
$$;

create or replace function public.complete_ingestion_job(
    p_job_id uuid,
    p_document_id uuid,
    p_tenant_id text,
    p_worker_id text,
    p_pages_processed integer,
    p_pages_requiring_ocr integer,
    p_chunks_created integer,
    p_vectors_upserted integer
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.ingestion_jobs j
    set status = 'completed',
        locked_at = null,
        locked_by = null,
        last_error = null,
        updated_at = now()
    where j.job_id = p_job_id
      and j.document_id = p_document_id
      and j.tenant_id = p_tenant_id
      and j.status = 'processing'
      and j.locked_by = p_worker_id;

    if not found then
        return false;
    end if;

    update public.documents d
    set status = 'completed',
        pages_processed = greatest(p_pages_processed, 0),
        pages_requiring_ocr = greatest(p_pages_requiring_ocr, 0),
        chunks_created = greatest(p_chunks_created, 0),
        vectors_upserted = greatest(p_vectors_upserted, 0),
        error_code = null,
        error_message = null,
        updated_at = now()
    where d.document_id = p_document_id
      and d.tenant_id = p_tenant_id;

    insert into public.document_audit_events (
        document_id,
        tenant_id,
        actor_id,
        action,
        metadata
    ) values (
        p_document_id,
        p_tenant_id,
        p_worker_id,
        'document.ingestion_completed',
        jsonb_build_object(
            'pages_processed', greatest(p_pages_processed, 0),
            'chunks_created', greatest(p_chunks_created, 0),
            'vectors_upserted', greatest(p_vectors_upserted, 0)
        )
    );

    return true;
end;
$$;

create or replace function public.fail_ingestion_job(
    p_job_id uuid,
    p_document_id uuid,
    p_tenant_id text,
    p_worker_id text,
    p_error_code text,
    p_error_message text,
    p_pages_processed integer,
    p_pages_requiring_ocr integer,
    p_chunks_created integer,
    p_vectors_upserted integer,
    p_retry_base_seconds integer
)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
    claimed_job public.ingestion_jobs%rowtype;
    next_status text;
    retry_delay_seconds integer;
begin
    select j.*
    into claimed_job
    from public.ingestion_jobs j
    where j.job_id = p_job_id
      and j.document_id = p_document_id
      and j.tenant_id = p_tenant_id
      and j.status = 'processing'
      and j.locked_by = p_worker_id
    for update;

    if not found then
        return 'stale';
    end if;

    if claimed_job.attempts < claimed_job.max_attempts then
        next_status := 'queued';
        retry_delay_seconds := greatest(p_retry_base_seconds, 1)
            * power(2, greatest(claimed_job.attempts - 1, 0))::integer;

        update public.ingestion_jobs j
        set status = next_status,
            available_at = now() + make_interval(secs => retry_delay_seconds),
            locked_at = null,
            locked_by = null,
            last_error = left(p_error_message, 2000),
            updated_at = now()
        where j.job_id = p_job_id;
    else
        next_status := 'failed';

        update public.ingestion_jobs j
        set status = next_status,
            locked_at = null,
            locked_by = null,
            last_error = left(p_error_message, 2000),
            updated_at = now()
        where j.job_id = p_job_id;
    end if;

    update public.documents d
    set status = next_status,
        pages_processed = greatest(p_pages_processed, 0),
        pages_requiring_ocr = greatest(p_pages_requiring_ocr, 0),
        chunks_created = greatest(p_chunks_created, 0),
        vectors_upserted = greatest(p_vectors_upserted, 0),
        error_code = left(p_error_code, 100),
        error_message = left(p_error_message, 1000),
        updated_at = now()
    where d.document_id = p_document_id
      and d.tenant_id = p_tenant_id;

    insert into public.document_audit_events (
        document_id,
        tenant_id,
        actor_id,
        action,
        metadata
    ) values (
        p_document_id,
        p_tenant_id,
        p_worker_id,
        case
            when next_status = 'queued' then 'document.ingestion_retry_scheduled'
            else 'document.ingestion_failed'
        end,
        jsonb_build_object(
            'attempt', claimed_job.attempts,
            'max_attempts', claimed_job.max_attempts,
            'error_code', left(p_error_code, 100)
        )
    );

    return next_status;
end;
$$;

revoke all on function public.claim_ingestion_job(text, integer)
    from public, anon, authenticated;
grant execute on function public.claim_ingestion_job(text, integer)
    to service_role;

revoke all on function public.complete_ingestion_job(
    uuid, uuid, text, text, integer, integer, integer, integer
) from public, anon, authenticated;
grant execute on function public.complete_ingestion_job(
    uuid, uuid, text, text, integer, integer, integer, integer
) to service_role;

revoke all on function public.fail_ingestion_job(
    uuid, uuid, text, text, text, text,
    integer, integer, integer, integer, integer
) from public, anon, authenticated;
grant execute on function public.fail_ingestion_job(
    uuid, uuid, text, text, text, text,
    integer, integer, integer, integer, integer
) to service_role;
