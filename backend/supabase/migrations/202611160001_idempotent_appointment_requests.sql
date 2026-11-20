create unique index appointment_requests_one_active_per_conversation_idx
    on public.appointment_requests (tenant_id, conversation_id)
    where status in ('collecting', 'awaiting_review', 'approved');

create or replace function public.get_or_create_appointment_request(
    p_appointment_request_id uuid,
    p_tenant_id text,
    p_conversation_id uuid,
    p_created_at timestamptz
)
returns setof public.appointment_requests
language plpgsql
security definer
set search_path = public
as $$
declare
    request_row public.appointment_requests%rowtype;
begin
    insert into public.appointment_requests (
        appointment_request_id,
        tenant_id,
        conversation_id,
        consent_given,
        status,
        created_at,
        updated_at
    )
    values (
        p_appointment_request_id,
        p_tenant_id,
        p_conversation_id,
        false,
        'collecting',
        p_created_at,
        p_created_at
    )
    on conflict (tenant_id, conversation_id)
        where status in ('collecting', 'awaiting_review', 'approved')
        do nothing;

    select *
    into request_row
    from public.appointment_requests
    where tenant_id = p_tenant_id
      and conversation_id = p_conversation_id
      and status in ('collecting', 'awaiting_review', 'approved')
    order by created_at
    limit 1;

    if not found then
        raise exception 'Unable to create appointment request.'
            using errcode = 'P0001';
    end if;

    return next request_row;
end;
$$;

revoke all on function public.get_or_create_appointment_request(
    uuid,
    text,
    uuid,
    timestamptz
) from public, anon, authenticated;

grant execute on function public.get_or_create_appointment_request(
    uuid,
    text,
    uuid,
    timestamptz
) to service_role;
