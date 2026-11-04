# Supabase setup

1. Apply the SQL migrations to the Supabase project.
2. Create a **private** Storage bucket named `medical-documents` (or use the
   value configured by `SUPABASE_DOCUMENT_BUCKET`).
3. Restrict the bucket to `application/pdf` and set its file-size limit to the
   same value as `MAX_UPLOAD_BYTES`.
4. Configure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` only in the backend
   deployment environment. Never expose the service-role key to a browser or
   commit it to source control.

The API stores objects below
`tenants/{tenant_id}/documents/{document_id}/original.pdf`. The database
function creates the document, ingestion job, outbox event, and audit event in
one transaction.

## Ingestion worker

Apply pending migrations before starting a worker:

```bash
supabase db push --linked --skip-vault
```

Run one worker locally with:

```bash
uv sync --locked
uv run python -m worker
```

The worker requires these environment variables in addition to the Google GenAI
and Pinecone variables already used by the ingestion pipeline:

```dotenv
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
SUPABASE_DOCUMENT_BUCKET=medical-documents
INGESTION_WORKER_POLL_SECONDS=5
INGESTION_JOB_LEASE_SECONDS=1800
INGESTION_RETRY_BASE_SECONDS=30
```

The API and worker are separate processes. FastAPI Cloud stores a queued job;
the OCI worker atomically claims it, downloads the private PDF, runs ingestion,
and updates the document to `completed`, `queued` for retry, or `failed`.
The active worker renews its lease every third of the configured lease period;
expired processing leases can be reclaimed after 30 minutes. Failures retry up
to the job's `max_attempts` with exponential delay. Google embedding rate
limits are retried inside the active job using the provider's requested delay,
so a long document does not restart from its first embedding batch.

## OCI systemd deployment

The unit in `deploy/medical-ingestion-worker.service` expects the checkout at
`/opt/medical-agentic-chatbot/backend` and secrets at
`/etc/medical-agentic-chatbot/worker.env`. The secret file must be owned by
`root:root`, mode `0600`, and must never be committed.

```bash
sudo cp deploy/medical-ingestion-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now medical-ingestion-worker
sudo systemctl status medical-ingestion-worker
sudo journalctl -u medical-ingestion-worker -f
```

The OCI E2 Micro runs a single worker because it has 1 GB RAM. Configure swap
before processing large PDFs. No inbound application port or public domain is
required; the worker only makes outbound HTTPS connections.
