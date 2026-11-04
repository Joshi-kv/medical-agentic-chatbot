# Supabase Ingestion Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume queued Supabase ingestion jobs and move uploaded documents to `completed` or `failed` after running the existing PDF-to-Pinecone pipeline.

**Architecture:** Supabase RPCs provide atomic job leasing and status transitions. A standalone asynchronous worker downloads one PDF at a time, validates it, invokes the synchronous ingestion pipeline in a thread, and persists its result before polling again.

**Tech Stack:** Python 3.12, Supabase Python SDK/Postgres, AnyIO, PyMuPDF, Google GenAI embeddings, Pinecone, pytest.

**Spec:** `docs/superpowers/specs/2026-11-03-supabase-ingestion-worker-design.md`

## Global Constraints

- Keep the FastAPI Cloud API process separate from the worker process.
- Run one worker process on the repurposed OCI E2 Micro VM with systemd and swap.
- Do not log or persist credentials, PDF bytes, or full exception traces.
- Use deterministic Pinecone IDs already produced by the chunker.
- Preserve tenant scoping for every database and vector operation.
- Do not commit automatically because the existing worktree contains uncommitted user changes.

## Review Focus

- Two workers racing for one job: exactly one claim succeeds.
- Worker termination during processing: the job becomes claimable after lease expiry.
- Corrupt or mismatched Storage object: no embeddings are written and failure is retried/recorded.
- Final attempt failure: both job and document become `failed` with safe error details.
- Empty queue: worker waits without issuing a hot polling loop.

---

### Task 1: Durable job leasing RPCs

**Files:**
- Create: `supabase/migrations/202611030002_ingestion_worker.sql`
- Test: `src/tests/unit/repositories/test_supabase_ingestion_jobs.py`
- Modify: `src/repositories/supabase_documents.py`

**Interfaces:**
- Produces: `IngestionJob`, `claim_ingestion_job(worker_id, lease_seconds)`, `complete_ingestion_job(...)`, and `fail_ingestion_job(...)` on `SupabaseDocumentStore`.

- [ ] Write failing adapter tests for empty claims, row mapping, completion parameters, retry status, and terminal failure status.
- [ ] Run `uv run pytest src/tests/unit/repositories/test_supabase_ingestion_jobs.py -q` and verify missing interfaces fail.
- [ ] Add the dataclass and repository methods that invoke the RPCs without exposing provider errors.
- [ ] Add SQL functions using `FOR UPDATE SKIP LOCKED`, lease expiry, attempt limits, tenant-scoped document updates, and restricted `service_role` execution.
- [ ] Run the repository tests and verify they pass.

### Task 2: Storage download and integrity validation

**Files:**
- Modify: `src/storage/supabase_documents.py`
- Test: `src/tests/unit/integrations/test_supabase_documents.py`
- Create: `src/worker_runtime.py`
- Test: `src/tests/unit/test_worker_runtime.py`

**Interfaces:**
- Consumes: `IngestionJob` from Task 1.
- Produces: `SupabaseDocumentStorage.download(storage_key: str) -> bytes` and `validate_document_content(job, content) -> None`.

- [ ] Write failing tests for Storage download, valid content, size mismatch, checksum mismatch, and invalid PDF signature.
- [ ] Run the focused tests and verify missing methods fail.
- [ ] Implement the download adapter and integrity validator.
- [ ] Run the focused tests and verify they pass.

### Task 3: One-job ingestion execution

**Files:**
- Modify: `src/worker_runtime.py`
- Test: `src/tests/unit/test_worker_runtime.py`

**Interfaces:**
- Consumes: repository leasing methods, Storage download, `SourceDocument`, and `ingest_pdf`.
- Produces: `IngestionWorker.run_once() -> bool`.

- [ ] Write failing tests proving an empty queue returns `False`, successful processing records all counters, pipeline failure calls retry/failure persistence, and unexpected exceptions are converted to safe failure details.
- [ ] Run the focused tests and verify they fail for missing behavior.
- [ ] Implement `run_once`, temporary-file lifecycle, `anyio.to_thread.run_sync`, and result persistence.
- [ ] Run the focused tests and verify they pass.

### Task 4: Continuous worker entrypoint and configuration

**Files:**
- Modify: `src/config/settings.py`
- Create: `src/worker.py`
- Modify: `.env.example`
- Test: `src/tests/unit/test_worker_entrypoint.py`

**Interfaces:**
- Consumes: `IngestionWorker.run_once() -> bool`.
- Produces: `build_ingestion_worker(settings) -> IngestionWorker`, `run_worker() -> None`, and executable `uv run python -m worker`.

- [ ] Write failing tests for service construction, configured polling, idle sleep, and cancellation propagation.
- [ ] Run the focused tests and verify missing entrypoint behavior fails.
- [ ] Add validated worker settings, construction, the polling loop, and concise structured logging.
- [ ] Run the focused tests and verify they pass.

### Task 5: Full verification and operator documentation

**Files:**
- Modify: `supabase/README.md`
- Create: `deploy/medical-ingestion-worker.service`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: local migration and worker commands, required environment-variable documentation, and an OCI systemd unit.

- [ ] Document `supabase db push`, `uv run python -m worker`, required environment, lease/retry behavior, and the separate-worker deployment boundary.
- [ ] Add a least-privilege systemd unit for `/opt/medical-agentic-chatbot/backend/.venv/bin/python -m worker` using `/etc/medical-agentic-chatbot/worker.env`.
- [ ] Run `uv run ruff check src` and require no findings.
- [ ] Run `uv run pytest -q` and require the full non-live suite to pass.
- [ ] Run `supabase db push --linked --dry-run --skip-vault` and verify only the new worker migration is pending before any remote apply.
- [ ] After local verification, stop and remove only the identified `avt-bridge` application, deploy the worker to the preserved VM, add swap if absent, and verify the systemd unit reaches `active` without printing secrets.
