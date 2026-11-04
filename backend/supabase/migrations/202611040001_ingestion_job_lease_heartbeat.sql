create or replace function public.renew_ingestion_job_lease(
    p_job_id uuid,
    p_worker_id text
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
begin
    update public.ingestion_jobs j
    set locked_at = now(),
        updated_at = now()
    where j.job_id = p_job_id
      and j.status = 'processing'
      and j.locked_by = p_worker_id;

    return found;
end;
$$;

revoke all on function public.renew_ingestion_job_lease(uuid, text)
    from public, anon, authenticated;
grant execute on function public.renew_ingestion_job_lease(uuid, text)
    to service_role;
