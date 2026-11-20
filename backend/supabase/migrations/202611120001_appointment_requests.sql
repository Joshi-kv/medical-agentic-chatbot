create table public.appointment_requests (
    appointment_request_id uuid primary key,
    tenant_id text not null,
    conversation_id uuid not null
        references public.conversations(conversation_id)
        on delete cascade,

    department text,
    preferred_date text,
    preferred_time text,
    contact text,
    reason text,
    consent_given boolean not null default false,

    status text not null default 'collecting'
        check (
            status in (
                'collecting',
                'awaiting_review',
                'approved',
                'rejected',
                'booked',
                'failed'
            )
        ),

    review_case_id uuid
        references public.human_review_cases(case_id),

    appointment_id uuid,
    error_message text,

    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index appointment_requests_tenant_conversation_idx
    on public.appointment_requests (
        tenant_id,
        conversation_id
    );

alter table public.appointment_requests enable row level security;

revoke all on public.appointment_requests
    from public, anon, authenticated;

grant select, insert, update
    on public.appointment_requests
    to service_role;

create table public.appointments (
    appointment_id uuid primary key,
    tenant_id text not null,
    appointment_request_id uuid not null unique
        references public.appointment_requests(appointment_request_id),

    department text not null,
    scheduled_date text not null,
    scheduled_time text,
    contact text not null,
    status text not null default 'scheduled'
        check (status in ('scheduled', 'cancelled')),

    created_at timestamptz not null default now()
);

alter table public.appointments enable row level security;

revoke all on public.appointments
    from public, anon, authenticated;

grant select, insert, update
    on public.appointments
    to service_role;
