# Supabase Ingestion Worker Design

## Intent

Uploaded medical PDFs currently remain `queued` because the API persists jobs but
no process consumes them. Add a standalone Python worker that uses Supabase
Postgres as a durable queue and completes the existing PDF-to-Pinecone pipeline.

## Architecture

FastAPI Cloud remains responsible only for HTTP upload and status endpoints. A
separately launched `python -m worker` process continuously claims jobs through
atomic Supabase RPCs. The worker downloads each private PDF from Supabase
Storage, verifies its size and SHA-256 checksum, runs the existing ingestion
pipeline in a thread, and records progress plus the terminal state in Postgres.

Job claims use `FOR UPDATE SKIP LOCKED`, a worker identity, attempts, and an
expiring lease. This permits multiple worker replicas and reclaims work after a
crashed worker. Pinecone chunk identifiers are deterministic, making retried
upserts idempotent.

## Behavior

- Claim only queued jobs whose `available_at` has arrived, plus abandoned
  processing jobs whose lease expired.
- Move the document to `processing` when claimed.
- Verify downloaded bytes against stored size and checksum before parsing.
- Preserve the current loader, normalizer, chunker, Google embedding, and
  Pinecone pipeline.
- Store all progress counters when processing completes or fails.
- Retry transient failures with database-backed exponential delay until
  `max_attempts`; mark both job and document `failed` after the final attempt.
- Never place PDF bytes, provider credentials, or full exception traces into
  queue payloads or logs.
- Stop gracefully on process interruption; an interrupted job is recovered
  after its lease expires.

## Configuration

- `INGESTION_WORKER_POLL_SECONDS=5`
- `INGESTION_JOB_LEASE_SECONDS=1800`
- `INGESTION_RETRY_BASE_SECONDS=30`
- Existing Supabase, Google GenAI, Pinecone, embedding, and namespace settings
  are shared with the API.

## Deployment Boundary

The worker command is intentionally independent from FastAPI startup. Production
runs it on the existing OCI `avt-bridge` E2 Micro VM, repurposed in place as
`medical-ingestion-worker`. A systemd unit starts it after reboot and restarts it
after failure. The host has 1 GB RAM, so deployment adds swap and runs one job at
a time. FastAPI Cloud continues to deploy the API and can scale it to zero
without interrupting the OCI worker.

## Non-goals

- Temporal, Celery, Redis, and FastAPI `BackgroundTasks`.
- OCR implementation for image-only pages.
- Automatic deletion processing; only document ingestion is covered.
