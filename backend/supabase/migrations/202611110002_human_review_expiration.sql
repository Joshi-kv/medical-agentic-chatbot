create or replace function public.expire_human_review_cases(
    p_tenant_id text
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    expired_count integer;
begin
    update public.human_review_cases
    set status = 'expired',
        updated_at = now()
    where tenant_id = p_tenant_id
      and status = 'pending'
      and expires_at <= now();

    get diagnostics expired_count = row_count;
    return expired_count;
end;
$$;

revoke all on function public.expire_human_review_cases(text)
    from public, anon, authenticated;

grant execute on function public.expire_human_review_cases(text)
    to service_role;
