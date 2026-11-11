# Agentic Medical Bot Roadmap

Created: 2026-11-05

Last updated: 2026-11-11

## Product direction

Build an evidence-first hospital digital front door. The agent should hold a
bounded intake conversation, ask only the questions required for routing,
provide cited education, identify the hospital's appropriate service, pause for
human review when required, and schedule an appointment only after explicit
patient confirmation.

The product must not claim to diagnose, prescribe, replace emergency services,
or make autonomous clinical decisions. Emergency behavior, routing authority,
human-review thresholds, appointment eligibility, and outbound-call policy are
hospital-owned rules rather than model decisions.

The target product supports:

- cited text answers over curated medical sources;
- bounded agentic retrieval and approved tools;
- diagrams, tables, scans, and images inside trusted documents;
- separately controlled patient-image intake;
- persistent conversations with explicit retention controls;
- policy-driven intake and clarification;
- routing through a hospital-owned department and service catalogue;
- separate clinical-triage and scheduling-staff HITL queues;
- appointment discovery, patient confirmation, booking, waitlisting,
  cancellation, and rescheduling;
- LiveKit voice sessions with captions and interruption;
- future inbound and consented outbound phone calls through SIP;
- warm transfer from the voice agent to hospital staff;
- an optional AI avatar after voice is stable;
- human escalation with acknowledged handoff state.

## Current baseline

As of 2026-11-11, the backend has PDF ingestion, normalized chunks, NVIDIA
embeddings/chat integration, Pinecone retrieval, evidence assembly, cited
answering, Supabase conversations and turn persistence, API-key tenant scope,
and SSE streaming with progress, answer deltas, citations, cancellation, and
safe error events. The next work starts at guardrails and the hospital agent
state; it should not rebuild the completed text pipeline.

## Architectural decision

Keep one repository and three primary independently deployable processes, with
an optional fourth worker for queued notifications and outbound calls:

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
├── hospital service catalogue and routing policy
├── bounded intake/question planner
├── retrieval and evidence assembly
├── one explicit agent state machine
├── typed, allow-listed tools
├── HITL pause/resume and work queues
├── scheduling orchestration and confirmation
├── answer and citation validation
└── conversation persistence
             │
Supabase + Pinecone + model providers

Independent workers using the shared application layer
├── OCI ingestion worker
├── LiveKit realtime agent worker
└── optional notification/call-dispatch worker
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
- **Hospital scheduling/EHR integration:** external system of record for
  services, practitioners, schedules, slots, and appointments. Local tables
  keep only the identifiers and audit state required for orchestration.
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
│   ├── routing.py
│   ├── prompts.py
│   ├── nodes/
│   │   ├── intake.py
│   │   ├── guardrails.py
│   │   ├── department.py
│   │   ├── human_review.py
│   │   ├── scheduling.py
│   │   └── answering.py
│   └── tools/
│       ├── retrieve.py
│       ├── source.py
│       ├── web_search.py
│       ├── service_catalogue.py
│       ├── appointment.py
│       └── escalation.py
├── guardrails/
│   ├── models.py
│   ├── input.py
│   ├── tools.py
│   ├── evidence.py
│   └── output.py
├── services/
│   ├── retrieval.py
│   ├── evidence.py
│   ├── answering.py
│   ├── citations.py
│   ├── conversation.py
│   ├── turns.py
│   ├── safety.py
│   ├── hospital_services.py
│   ├── human_review.py
│   ├── scheduling.py
│   └── attachments.py
├── ingestion/
│   ├── pipeline.py
│   ├── ocr.py
│   ├── layout.py
│   ├── images.py
│   └── tables.py
├── integrations/
│   ├── livekit.py
│   ├── hospital_scheduler.py
│   ├── fhir.py
│   ├── sip.py
│   ├── speech.py
│   └── existing provider clients
├── realtime/
│   ├── worker.py
│   ├── session.py
│   ├── telephony.py
│   └── avatar.py
└── api/
    ├── routers/
    │   ├── conversations.py
    │   ├── chat.py
    │   ├── human_review.py
    │   ├── appointments.py
    │   ├── attachments.py
    │   └── realtime.py
    └── schemas/
        ├── conversations.py
        ├── chat.py
        ├── human_review.py
        ├── appointments.py
        ├── attachments.py
        └── realtime.py
```

Use LangGraph only for the explicit agent state machine. Keep normal business
logic in services so text and LiveKit sessions reuse the same implementation.
Use a durable checkpointer for HITL. Supabase conversation records remain the
long-term product memory; graph checkpoints are resumable execution state and
must not become an uncontrolled duplicate clinical record.

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
- `safety_events` and `escalation_cases`;
- `healthcare_services`: hospital-approved department/service directory;
- `department_routing_rules`: versioned eligibility and escalation policy;
- `intake_sessions`: collected fields, missing fields, consent, and status;
- `human_review_cases`: scheduling or clinical queue, SLA, owner, and decision;
- `agent_checkpoints`: durable LangGraph pause/resume state, encrypted and
  retention-controlled;
- `appointment_requests`: external IDs, proposed slot, confirmation and status;
- `appointment_actions`: idempotent booking/reschedule/cancel audit trail;
- `realtime_sessions` and `call_sessions`: channel, consent, transfer, outcome,
  and minimal provider identifiers.

The hospital scheduler or FHIR server remains authoritative for availability
and appointments. Do not copy full schedules or clinical records into local
tables without a defined need, access policy, and retention policy.

Store stable identifiers and required filter fields in Pinecone metadata. Keep
the tenant filter mandatory. Migrate to tenant-specific namespaces only with a
tested reindex plan.

## Agent flow

```text
authenticate tenant and patient/session
→ obtain channel-specific consent and determine the patient's goal
→ deterministic emergency/red-flag guardrail
    → emergency route: approved immediate guidance and notification; never wait
      indefinitely for HITL
→ load the minimum required intake fields for the requested service
→ ask one relevant clarification at a time with a question/time limit
→ classify the administrative or educational intent
    ├── education: run evidence-first cited answering
    ├── navigation: route only through the hospital service catalogue
    ├── high risk/low confidence: pause for clinical or scheduling HITL
    └── unsupported: explain boundary and offer staff handoff
→ after route approval, query schedules/slots from the hospital system
→ present choices and require explicit patient confirmation
→ create an idempotent proposed appointment request
→ confirm, waitlist, reschedule, or hand off based on the scheduler response
→ persist the turn, decision provenance, consent, and restricted audit record
→ stream text, speak in an app session, or later handle a SIP phone call
```

The model may interpret natural language and select only typed, allow-listed
tools. Tenant scope, emergency behavior, service eligibility, routing
thresholds, appointment confirmation, timeouts, evidence thresholds, and
handoff rules are not model decisions.

## Guardrail and authority model

Guardrails apply at every boundary rather than as one safety prompt:

- **Input:** identity, consent, prompt-injection resistance, supported intent,
  emergency/red-flag policy, and minimum necessary intake.
- **Tool:** tenant-bound arguments, approved services/domains, timeouts, result
  limits, idempotency, and least-privilege credentials.
- **Evidence:** provenance, freshness, score thresholds, tenant isolation,
  conflict handling, and citation resolvability.
- **Action:** department eligibility, human-review threshold, explicit patient
  confirmation, slot recheck, and duplicate-booking prevention.
- **Output:** supported claims, citations, no diagnosis/prescription, emergency
  wording, and discard of an unvalidated streamed draft.

Hard rules use deterministic code or a policy table. A model classifier may
assist with ambiguous language but cannot override a hard rule.

## Human-in-the-loop model

Use two initial queues:

```text
scheduling review
├── unclear department
├── eligibility/referral/insurance question
├── unavailable or conflicting resources
└── reschedule/cancellation exception

clinical triage review
├── high-risk or ambiguous symptoms
├── medication/treatment request
├── unclear urgency
└── hospital-policy escalation
```

The reviewer sees facts, patient answers, risk flags, candidate services,
confidence, and the requested action, never private model reasoning. Supported
decisions are `approve`, `redirect`, `ask_patient`, `urgent_escalation`, and
`reject`. A LangGraph interrupt pauses with a typed JSON payload and resumes
with the same stable thread ID. Every queue needs ownership, SLA, acknowledgement,
expiry, reassignment, and an offline fallback.

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
- `src/services/conversation.py`
- `src/services/turns.py`
- new chat/conversation routers and schemas

### Gate

Clinician-reviewed thresholds for retrieval recall, citation coverage and
entailment, unsupported claims, abstention, tenant isolation, and emergency
false negatives must pass.

## Milestone 3 — Hospital front door, guardrails, and bounded agentic RAG

This is the next active milestone. Build one policy-bounded graph that supports
education, navigation, clarification, staff review, and scheduling without
turning the model into an autonomous triage or booking authority.

### Tools

- `search_knowledge_base`
- `get_source_section`
- `compare_sources`
- `search_approved_web_sources`
- `inspect_document_image`
- `request_clarification`
- `trigger_emergency_guidance`
- `request_human_handoff`
- `list_hospital_services`
- `get_service_requirements`
- `find_available_slots`
- `propose_appointment`
- `confirm_appointment`
- `request_waitlist`

Every tool requires typed input/output, trusted tenant context, timeout, retry
limit, result-size limit, provenance, and an audit event. Tavily results remain
clearly labeled current-web evidence and cannot silently override curated
sources. Scheduling tools must require explicit confirmation and idempotency.

### State machine

```text
identify goal and consent
→ deterministic emergency policy
    ├── emergency: immediate approved response; optional notification
    └── continue
→ load service-specific intake requirements
→ collect missing information one question at a time
→ classify route
    ├── cited education
    ├── hospital navigation
    └── unsupported/high risk
→ evaluate risk and routing confidence
    ├── approved automatic route
    ├── scheduling HITL interrupt
    └── clinical HITL interrupt
→ resume with typed staff decision
→ query service schedule and slots
→ ask patient to explicitly confirm one proposed slot
→ recheck slot and perform idempotent booking
→ return confirmed, proposed, waitlisted, or handed-off result
```

The question planner asks only for fields required by the relevant hospital
service, stops when an emergency rule fires, avoids facts already present in
the conversation, and caps clarification attempts before staff handoff.

### Initial API surface

```text
POST /v1/conversations/{conversation_id}/turns/stream
GET  /v1/hospital-services
GET  /v1/human-review/cases
POST /v1/human-review/cases/{case_id}/claim
POST /v1/agent/runs/{run_id}/resume
GET  /v1/appointments/availability
POST /v1/appointments/proposals
POST /v1/appointments/{appointment_id}/confirm
POST /v1/appointments/{appointment_id}/cancel
POST /v1/appointments/{appointment_id}/reschedule
```

Do not expose internal graph state or private model reasoning. HITL payloads
contain structured facts, flags, candidates, provenance, and the requested
decision.

Conversation memory should contain recent turns plus an explicit stored
summary. Do not convert conversation text into permanent medical memory without
consent, provenance, correction, expiration, and deletion controls.

### Gate

Adversarial tests must show that user prompts, documents, web content, and tool
outputs cannot override policy, cross tenant boundaries, fabricate citations
or handoffs, or trigger unapproved actions. Scheduling tests must prove that
retries cannot create duplicate appointments, stale slots are rechecked,
booking requires explicit confirmation, emergencies do not wait indefinitely
for staff, and every interrupt can resume or expire cleanly.

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

Build in-app/browser voice before phone calls or avatar video.

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

## Milestone 7 — Inbound and outbound voice calling

Add telephony only after text and in-app voice use the same tested guardrails,
HITL, and scheduling runtime.

### Inbound calls

```text
hospital phone number or SIP trunk
→ LiveKit SIP participant and explicit agent dispatch
→ identify the caller without trusting caller ID alone
→ disclose that the caller is speaking with an AI agent
→ obtain recording/transcription consent where required
→ run the shared intake/guardrail/HITL/scheduling graph
→ warm-transfer to staff or create a callback case when required
```

Support DTMF fallback, poor-audio recovery, language selection, silence and
hang-up handling, staff transfer, callback creation, and a non-AI fallback.

### Outbound calls

Limit the first release to explicit, policy-approved purposes such as a
patient-requested callback, appointment confirmation, rescheduling, or
reminder. Require verified consent, permitted calling hours, retry limits,
do-not-call/suppression handling, approved scripts, and idempotent dispatch.
Never let the model invent a destination number or initiate unsolicited
clinical outreach.

Use a regional SIP provider when LiveKit-managed numbers do not cover the
deployment country. Keep phone numbers and SIP credentials out of model state,
prompts, ordinary logs, and client metadata.

### Gate

Pass simulated and real test calls for identity, consent, emergency behavior,
DTMF, interruption, noisy audio, transfer, voicemail, busy/no-answer, carrier
failure, reconnect, duplicate-call prevention, transcript retention, and
staff/callback fallback. Outbound calling requires hospital legal, privacy,
and communications-policy approval.

## Milestone 8 — Live avatar

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

## Milestone 9 — Controlled pilot

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
- department-routing accuracy and unnecessary-question rate;
- HITL queue wait, acknowledgement, expiry, redirect, and resolution rate;
- booking success, duplicate prevention, stale-slot conflict, and cancellation;
- voice transcription accuracy, transfer success, call abandonment, and
  recovery by language and channel;
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

LiveKit milestones additionally require browser E2E tests for join/token
expiry, audio permissions, captions, interruption, reconnect, worker loss,
text fallback, avatar loss, and session cleanup. Telephony additionally needs
SIP-provider and LiveKit artifact verification: trunks, dispatch rules, phone
numbers, transfer targets, caller-ID policy, recording configuration, and the
deployed realtime worker version.

## What not to build yet

- No autonomous multi-agent swarm.
- No diagnosis, prescription generation, or autonomous treatment selection.
- No EHR writes or irreversible clinical actions.
- No autonomous appointment booking without explicit patient confirmation.
- No model-generated department or service outside the approved catalogue.
- No indefinite HITL wait on an emergency route.
- No unsolicited outbound calling or model-selected phone numbers.
- No unrestricted open-web medical search.
- No permanent medical memory without explicit consent and controls.
- No patient-image diagnosis before trusted-document multimodal retrieval is
  validated.
- No avatar before text and voice safety are stable.
- No self-hosted LiveKit SFU or large vision/LLM on the free OCI VM.
- No model fine-tuning before evaluation data identifies a concrete gap.
- No production dependency on a free preview model endpoint without an SLA.

## Immediate next sprint

1. Define `HospitalAgentState`, typed route/disposition enums, and stable graph
   thread/run identity without storing prompts or private reasoning in state.
2. Create the versioned hospital service catalogue and routing-policy schema;
   seed a small non-production catalogue for tests.
3. Implement deterministic input, tool, evidence, action, and output guardrail
   interfaces with an explicit allow/escalate/block decision.
4. Implement the intake/question planner with required fields, one-question
   turns, no-repeat behavior, and a clarification cap.
5. Build department candidate lookup against the catalogue and a confidence
   gate; do not add real appointment writes yet.
6. Add durable LangGraph checkpointing, typed scheduling/clinical interrupts,
   review-case persistence, claim/expiry behavior, and a resume endpoint.
7. Add a mock scheduling provider and prove confirmation, idempotency, stale
   slot handling, waitlist, cancellation, and rescheduling end to end.
8. Reuse the existing SSE contract for `agent_status`, `answer_delta`,
   `citations`, `turn_completed`, `interrupt`, and `error` events.

Do not begin LiveKit, SIP, avatar, or real hospital scheduler writes during
this sprint. Promote the mock provider to a real FHIR/vendor integration only
after routing and HITL gates pass.

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
- LiveKit telephony and SIP:
  https://docs.livekit.io/telephony/
- LiveKit outbound calls:
  https://docs.livekit.io/telephony/making-calls/outbound-calls/
- LiveKit avatars:
  https://docs.livekit.io/agents/models/avatar/
- LangGraph interrupts:
  https://docs.langchain.com/oss/python/langgraph/interrupts
- HL7 FHIR HealthcareService:
  https://hl7.org/fhir/R4/healthcareservice.html
- HL7 FHIR Appointment:
  https://hl7.org/fhir/R4/appointment.html
