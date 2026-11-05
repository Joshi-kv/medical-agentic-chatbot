# Agentic Medical Bot Roadmap

Date: 2026-11-05

## Product direction

Build an evidence-first medical education and navigation assistant. The initial
product must not claim to diagnose, prescribe, replace emergency services, or
make autonomous clinical decisions.

The target product supports:

- cited text answers over curated medical sources;
- bounded agentic retrieval and approved tools;
- diagrams, tables, scans, and images inside trusted documents;
- separately controlled patient-image intake;
- persistent conversations with explicit retention controls;
- LiveKit voice sessions with captions and interruption;
- an optional AI avatar after voice is stable;
- human escalation with acknowledged handoff state.

## Architectural decision

Keep one repository and three independently deployable processes:

```text
Web/mobile client
├── text, citations, uploads and conversation history
└── LiveKit WebRTC audio/video
             │
FastAPI control plane
├── authentication and trusted tenant context
├── documents, attachments and conversations
├── text chat/SSE streaming
├── retrieval and citation APIs
└── short-lived LiveKit room tokens
             │
Shared application layer
├── safety and triage policy
├── retrieval and evidence assembly
├── one explicit agent state machine
├── typed, allow-listed tools
├── answer and citation validation
└── conversation persistence
             │
Supabase + Pinecone + model providers

Independent workers using the shared application layer
├── OCI ingestion worker
└── LiveKit realtime agent worker
```

Do not create a multi-agent swarm. “Agentic” should mean conditional routing,
bounded tool use, evidence checks, clarification, abstention, and escalation.

## Runtime boundaries

- **FastAPI Cloud:** authenticated HTTP/SSE APIs, conversations, uploads,
  retrieval, LiveKit session/token creation.
- **OCI ingestion worker:** durable extraction, OCR, chunking, embedding, and
  indexing. It must not host realtime voice sessions.
- **LiveKit agent worker:** STT, turn detection, shared agent runtime, and TTS.
  It must not run inside FastAPI's request lifecycle.
- **Supabase:** authoritative records, private assets, conversations, jobs,
  citations, consent, and audit events.
- **Pinecone:** rebuildable derived indexes, never the system of record.
- **Model providers:** replaceable integrations, never the source of medical
  policy.

## Target source layout

Extend the existing project without replacing its current structure:

```text
backend/src/
├── agent/
│   ├── state.py
│   ├── graph.py
│   ├── prompts.py
│   └── tools/
│       ├── retrieve.py
│       ├── source.py
│       ├── web_search.py
│       └── escalation.py
├── services/
│   ├── retrieval.py
│   ├── evidence.py
│   ├── answering.py
│   ├── citations.py
│   ├── conversations.py
│   ├── safety.py
│   └── attachments.py
├── ingestion/
│   ├── pipeline.py
│   ├── ocr.py
│   ├── layout.py
│   ├── images.py
│   └── tables.py
├── integrations/
│   ├── livekit.py
│   ├── speech.py
│   └── existing provider clients
├── realtime/
│   ├── worker.py
│   ├── session.py
│   └── avatar.py
└── api/
    ├── routers/
    │   ├── conversations.py
    │   ├── chat.py
    │   ├── attachments.py
    │   └── realtime.py
    └── schemas/
        ├── conversations.py
        ├── chat.py
        ├── attachments.py
        └── realtime.py
```

Use LangGraph only for the explicit agent state machine. Keep normal business
logic in services so text and LiveKit sessions reuse the same implementation.

## Data model direction

Add Supabase migrations incrementally for:

- `source_collections`: tenant-owned curated corpora;
- `source_versions`: publisher, authority, jurisdiction, population, effective
  and review dates, license, checksum, and approval status;
- `document_pages`: text, OCR/layout status, page-image object key;
- `content_chunks`: stable span, page, section, modality, source version;
- `conversations` and `messages`;
- `turn_runs`: routing, policy, model/prompt version, latency, disposition;
- `citations`: claim-to-source-span mapping;
- `attachments`: consent, MIME, checksum, EXIF removal, retention state;
- `tool_runs`: typed input/output references and status;
- `safety_events` and `escalation_cases`.

Store stable identifiers and required filter fields in Pinecone metadata. Keep
the tenant filter mandatory. Migrate to tenant-specific namespaces only with a
tested reindex plan.

## Agent flow

```text
authenticate tenant and user
→ normalize text and attachments
→ deterministic emergency/red-flag policy
    → emergency route: concise action and escalation, no long agent loop
→ classify supported intent
    → unsupported/high-risk: clarify, abstain, or escalate
→ select evidence route
    ├── curated text retrieval
    ├── trusted-document image/table retrieval
    ├── limited patient-image observation
    └── approved current-web research
→ retrieve, fuse, rerank, and apply evidence threshold
→ compose structured answer with per-claim citations
→ validate citations and post-generation safety
→ persist turn and restricted audit record
→ stream text or speak through LiveKit
```

The model may select only typed, allow-listed tools. Tenant scope, safety
routes, timeouts, evidence thresholds, and handoff rules are not model
decisions.

## Milestone 0 — Safety and current-system cleanup

### Work

- Define intended use, prohibited behavior, operating jurisdiction, and a
  clinical safety owner.
- Define emergency/red-flag and human-escalation policies.
- Replace the shared API key with user/session authentication before patient
  use; rotate all exposed credentials immediately.
- Define consent, retention, deletion, vendor, logging, and PHI policies.
- Add retrieval route tests and a clinician-reviewed evaluation harness.
- Align `document_id` types across API and retrieval service.
- Remove duplicate vectors/documents and add duplicate-content handling.
- Introduce a versioned source registry before treating content as clinical
  evidence.

### Gate

No patient-facing release until intended use, emergency behavior, privacy
boundaries, tenant isolation, clinical review ownership, and vendor handling
are documented and tested.

## Milestone 1 — Durable evidence ingestion

Multimodal jobs will be longer than the current PDF jobs, so improve durability
before adding modalities.

### Work

- Persist chunk-level checkpoints instead of embedding the full document before
  any upsert.
- Upsert incrementally and resume from the last completed chunk.
- Track extraction, OCR, embedding, and upsert progress separately.
- Add source/version IDs to every page, chunk, and vector.
- Add delete, reindex, retirement, and admin retry workflows.
- Deduplicate by tenant, checksum, source version, model, and pipeline version.
- Add ingestion health, latency, and cost metrics.

### Critical files

- `src/ingestion/pipeline.py`
- `src/worker_runtime.py`
- `src/services/documents.py`
- `src/repositories/supabase_documents.py`
- new Supabase migrations

### Gate

A worker restart must resume without regenerating completed vectors, and
delete/reindex operations must leave Supabase and Pinecone consistent.

## Milestone 2 — Evidence-first text assistant

Build cited answering before autonomous tool selection.

### Work

- Persist conversations, messages, runs, and citations.
- Add candidate retrieval, passage deduplication, minimum evidence thresholds,
  and optional reranking.
- Add lexical/hybrid search later for exact drug names, abbreviations, and
  medical codes after measuring dense retrieval gaps.
- Generate answers only from an explicit evidence package.
- Validate that every rendered citation resolves to the retrieved source span.
- Abstain or clarify when evidence is weak, stale, missing, or conflicting.
- Stream text with SSE before introducing realtime audio.

### Endpoints

```text
POST /v1/conversations
GET  /v1/conversations/{conversation_id}
GET  /v1/conversations/{conversation_id}/messages
POST /v1/conversations/{conversation_id}/turns
```

### Response contract

```text
answer
urgency
limitations
claims[]
citations[]
unknowns[]
follow_up_question
abstained
recommended_disposition
```

### Critical files

- `src/services/retrieval.py`
- new `src/services/evidence.py`
- new `src/services/answering.py`
- new `src/services/citations.py`
- new `src/services/conversations.py`
- new chat/conversation routers and schemas

### Gate

Clinician-reviewed thresholds for retrieval recall, citation coverage and
entailment, unsupported claims, abstention, tenant isolation, and emergency
false negatives must pass.

## Milestone 3 — Bounded agentic RAG

### Tools

- `search_knowledge_base`
- `get_source_section`
- `compare_sources`
- `search_approved_web_sources`
- `inspect_document_image`
- `request_clarification`
- `trigger_emergency_guidance`
- `request_human_handoff`

Every tool requires typed input/output, trusted tenant context, timeout, retry
limit, result-size limit, provenance, and an audit event. Tavily results remain
clearly labeled current-web evidence and cannot silently override curated
sources.

### State machine

```text
intake
→ deterministic safety route
→ clarify or plan
→ retrieve/approved tool calls
→ evidence sufficiency
→ answer
→ citation validation
→ post-answer safety
→ respond or hand off
```

Conversation memory should contain recent turns plus an explicit stored
summary. Do not convert conversation text into permanent medical memory without
consent, provenance, correction, expiration, and deletion controls.

### Gate

Adversarial tests must show that user prompts, documents, web content, and tool
outputs cannot override policy, cross tenant boundaries, fabricate citations
or handoffs, or trigger unapproved actions.

## Milestone 4 — Multimodal trusted-document RAG

Start with images, scans, diagrams, and tables inside approved medical sources,
not patient photos.

### Ingestion

```text
PDF page
├── extracted text
├── OCR text
├── layout regions and coordinates
├── tables and structured cells
├── figures/images and captions
└── private page image
```

Store the original page image and all derived artifacts in private Supabase
storage with source-version, page, and bounding-box links.

### Indexing and retrieval

- Keep text, OCR/layout text, tables, and visual embeddings as explicit
  modalities.
- Never mix unrelated embedding models in one vector space. Use separate
  indexes/namespaces where required.
- Retrieve modality candidates independently, apply rank fusion, then rerank.
- Return both exact text spans and page-image regions for visual citations.

Suggested vector metadata:

```text
tenant_id
source_version_id
document_id
page_number
chunk_id
modality: text | page_image | table
section
effective_date
review_status
```

### Gate

Benchmark text, scan/OCR, table, chart, and diagram retrieval independently and
together. Answers must point to the correct visual region and abstain when a
visual cannot be read reliably.

## Milestone 5 — Controlled patient-image intake

Keep user images out of the general knowledge index.

```text
explicit consent
→ private upload and malware/type/size validation
→ remove EXIF/location
→ image quality/usability check
→ collect body site, duration, symptoms and context
→ limited observation extraction
→ retrieve approved visual references
→ cautious response or acknowledged human review
→ retention/deletion enforcement
```

Limit claims to a clinically and regulatorily reviewed use case. Do not present
unsupported definitive diagnoses. Poor-quality and high-risk cases must force
clarification or escalation.

### Gate

Clinical/regulatory approval, subgroup evaluation including skin tone and
device/image quality, explicit consent, retention controls, and a reliable
human-review workflow are required.

## Milestone 6 — LiveKit voice agent

Build voice before avatar video.

### FastAPI responsibility

```text
POST /v1/realtime/sessions
→ authenticate user
→ create/attach conversation
→ create and dispatch LiveKit room
→ issue short-lived room-scoped JWT
→ return LiveKit URL, token and conversation ID
```

Tenant/user identity must come from the authenticated server session mapping,
not client room metadata.

### Realtime worker

Start with an auditable cascaded pipeline:

```text
STT → canonical transcript → shared agent runtime → TTS
```

Reuse the same safety, retrieval, citations, memory, audit, and escalation
services as text chat. Add captions before avatar work. Support interruption,
false-barge handling, endpoint tuning, and text fallback. Evaluate English and
Hindi medical terminology if both languages are in scope.

Track STT delay, end-of-turn delay, LLM first-token time, TTS first-byte time,
total latency, interruption errors, and cost per minute.

### Gate

Voice must pass the same medical safety/evidence evaluation as text, retain an
auditable transcript, and fall back safely if any realtime component fails.

## Milestone 7 — Live avatar

Use a LiveKit-supported avatar provider initially. The provider joins as a
room participant and publishes synchronized audio/video; the frontend renders
standard LiveKit tracks.

Evaluate providers for latency, lip sync, language/voice support, regional
availability, retention/training terms, contractual PHI support, cost, and
fallback behavior. Keep audio-only mode. Clearly identify the avatar as AI and
require explicit consent.

### Gate

Do not pilot an avatar until voice meets latency and safety thresholds, privacy
review is complete, audio fallback works, and avatar failure cannot interrupt
emergency instructions or human handoff.

## Milestone 8 — Controlled pilot

- Internal and clinician shadow mode.
- Narrow intended use and limited cohort.
- Human review of sampled and flagged sessions.
- Incident response, rollback, and provider fallback procedures.
- Drift monitoring after every model, prompt, source, policy, or tool change.
- Prospective safety and subgroup monitoring before expanding indications.

## Evaluation and observability

Maintain a locked, clinician-reviewed suite covering routine questions,
ambiguity, conflicting evidence, pregnancy/pediatric cases, rare high-risk
events, emergency symptoms, prompt injection, poisoned sources, cross-tenant
access, provider outages, visual quality, and out-of-distribution inputs.

Measure:

- emergency sensitivity and false-negative rate;
- retrieval recall@k and source freshness;
- citation correctness, entailment, and claim coverage;
- unsupported/unsafe advice rate;
- abstention and clarification quality;
- image performance across relevant subgroups and capture conditions;
- escalation acknowledgement and failure;
- latency, availability, and cost by stage.

Trace request/run IDs, route decisions, policy flags, provider/model/prompt
versions, retrieved source IDs, citation checks, tool calls, latency, and
disposition. Keep raw PHI out of ordinary logs and analytics; use encrypted,
restricted audit records where necessary.

## Deployment verification

For each milestone:

1. Run Ruff and the full unit suite.
2. Run integration tests against isolated Supabase/Pinecone namespaces.
3. Run tenant-isolation and authorization tests.
4. Run locked medical safety/evidence evaluations.
5. Test provider timeout, 429, 5xx, partial failure, and fallback behavior.
6. Verify the actual deployed artifacts, environment, entrypoint, service
   process, model/index dimensions, and remote image/container digest.
7. Release first in shadow/internal mode with rollback controls.

LiveKit milestones additionally require browser E2E tests for join/token expiry,
audio permissions, captions, interruption, reconnect, worker loss, text
fallback, avatar loss, and session cleanup.

## What not to build yet

- No autonomous multi-agent swarm.
- No diagnosis, prescription generation, or autonomous treatment selection.
- No EHR writes or irreversible clinical actions.
- No unrestricted open-web medical search.
- No permanent medical memory without explicit consent and controls.
- No patient-image diagnosis before trusted-document multimodal retrieval is
  validated.
- No avatar before text and voice safety are stable.
- No self-hosted LiveKit SFU or large vision/LLM on the free OCI VM.
- No model fine-tuning before evaluation data identifies a concrete gap.
- No production dependency on a free preview model endpoint without an SLA.

## Immediate next sprint

1. Finish retrieval route tests and normalize `document_id` typing.
2. Remove duplicate book vectors and add checksum-based duplicate protection.
3. Create the source/version schema and attach provenance to new chunks.
4. Create the clinician-reviewed retrieval/evidence evaluation harness.
5. Add conversation/message migrations and repository interfaces.
6. Implement `EvidenceService`, then cited `AnsweringService`.
7. Expose an SSE text-turn endpoint and validate citations/abstention.

Do not begin LiveKit or avatar integration during this sprint.

## Reference guidance

- FDA, Clinical Decision Support Software guidance:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- HHS, HIPAA and cloud computing:
  https://www.hhs.gov/hipaa/for-professionals/special-topics/health-information-technology/cloud-computing/index.html
- HHS, HIPAA Security Rule:
  https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html
- NIST AI RMF 1.0:
  https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10
- NIST Generative AI Profile:
  https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
- WHO, ethics and governance of AI for health:
  https://www.who.int/publications/b/58847
- LiveKit pipeline types:
  https://docs.livekit.io/agents/models/pipelines/
- LiveKit RAG/external data:
  https://docs.livekit.io/agents/logic/external-data/
- LiveKit authentication and tokens:
  https://docs.livekit.io/home/concepts/authentication/
- LiveKit avatars:
  https://docs.livekit.io/agents/models/avatar/
