create or replace function public.book_appointment_request(
    p_appointment_request_id uuid,
    p_tenant_id text,
    p_appointment_id uuid,
    p_created_at timestamptz
)
returns setof public.appointments
language plpgsql
security definer
set search_path = public
as $$
declare
    request_row public.appointment_requests%rowtype;
    appointment_row public.appointments%rowtype;
begin
    select *
    into request_row
    from public.appointment_requests
    where appointment_request_id = p_appointment_request_id
      and tenant_id = p_tenant_id
    for update;

    if not found then
        raise exception 'Appointment request not found.'
            using errcode = 'P0002';
    end if;

    select *
    into appointment_row
    from public.appointments
    where appointment_request_id = p_appointment_request_id
      and tenant_id = p_tenant_id;

    if found then
        update public.appointment_requests
        set status = 'booked',
            appointment_id = appointment_row.appointment_id,
            updated_at = now()
        where appointment_request_id = p_appointment_request_id
          and tenant_id = p_tenant_id;

        return next appointment_row;
        return;
    end if;

    if request_row.status <> 'approved' then
        raise exception 'Appointment request is not approved.'
            using errcode = 'P0001';
    end if;

    if request_row.department is null
       or btrim(request_row.department) = ''
       or request_row.preferred_date is null
       or btrim(request_row.preferred_date) = ''
       or request_row.contact is null
       or btrim(request_row.contact) = ''
       or request_row.consent_given is not true then
        raise exception 'Appointment request is incomplete.'
            using errcode = 'P0001';
    end if;

    insert into public.appointments (
        appointment_id,
        tenant_id,
        appointment_request_id,
        department,
        scheduled_date,
        scheduled_time,
        contact,
        status,
        created_at
    )
    values (
        p_appointment_id,
        p_tenant_id,
        p_appointment_request_id,
        request_row.department,
        request_row.preferred_date,
        request_row.preferred_time,
        request_row.contact,
        'scheduled',
        p_created_at
    )
    returning * into appointment_row;

    update public.appointment_requests
    set status = 'booked',
        appointment_id = appointment_row.appointment_id,
        updated_at = now()
    where appointment_request_id = p_appointment_request_id
      and tenant_id = p_tenant_id;

    return next appointment_row;
end;
$$;

revoke all on function public.book_appointment_request(
    uuid,
    text,
    uuid,
    timestamptz
) from public, anon, authenticated;

grant execute on function public.book_appointment_request(
    uuid,
    text,
    uuid,
    timestamptz
) to service_role;
