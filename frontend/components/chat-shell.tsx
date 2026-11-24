"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { readSseStream } from "@/lib/sse";
import type { Citation, Conversation, ReviewCase, StoredMessage } from "@/lib/types";

const CONVERSATION_KEY = "pulse-care-conversation";

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  safetyNotice?: string;
  reviewCaseId?: string | null;
};

type AgentStatus = { stage: string; message: string };

function messageFromStored(message: StoredMessage): ChatMessage {
  return { id: message.message_id, role: message.role, content: message.content };
}

function errorMessage(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "Something went wrong. Please try again.";
  const detail = "detail" in payload ? payload.detail : null;
  if (detail && typeof detail === "object" && "message" in detail && typeof detail.message === "string") {
    return detail.message;
  }
  return "Something went wrong. Please try again.";
}

async function jsonOrNull(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export function ChatShell() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [agentStatus, setAgentStatus] = useState<AgentStatus | null>(null);
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const threadEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const savedId = window.localStorage.getItem(CONVERSATION_KEY);
    if (!savedId) return;

    setConversationId(savedId);
    fetch(`/api/conversations/${savedId}/messages`)
      .then(async (response) => {
        if (!response.ok) throw new Error(errorMessage(await jsonOrNull(response)));
        return (await response.json()) as StoredMessage[];
      })
      .then((stored) => setMessages(stored.map(messageFromStored)))
      .catch(() => {
        window.localStorage.removeItem(CONVERSATION_KEY);
        setConversationId(null);
        setLoadError("Your previous conversation could not be restored.");
      });
  }, []);

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, agentStatus]);

  async function ensureConversation(): Promise<string> {
    if (conversationId) return conversationId;

    const response = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: null }),
    });
    const payload = await jsonOrNull(response);
    if (!response.ok) throw new Error(errorMessage(payload));

    const conversation = payload as Conversation;
    setConversationId(conversation.conversation_id);
    window.localStorage.setItem(CONVERSATION_KEY, conversation.conversation_id);
    return conversation.conversation_id;
  }

  function startNewConversation() {
    if (isLoading) return;
    window.localStorage.removeItem(CONVERSATION_KEY);
    setConversationId(null);
    setMessages([]);
    setAgentStatus(null);
    setCompletedStages([]);
    setLoadError(null);
  }

  async function changeAccessKey() {
    await fetch("/api/access/logout", { method: "POST" });
    window.localStorage.removeItem(CONVERSATION_KEY);
    window.location.assign("/access");
  }

  async function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const query = question.trim();
    if (!query || isLoading) return;

    setQuestion("");
    setIsLoading(true);
    setLoadError(null);
    setCompletedStages([]);

    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", content: query };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", content: "" },
    ]);

    try {
      const id = await ensureConversation();
      const response = await fetch(`/api/conversations/${id}/turns/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!response.ok || !response.body) {
        throw new Error(errorMessage(await jsonOrNull(response)));
      }

      for await (const streamEvent of readSseStream(response.body)) {
        if (streamEvent.event === "agent_status") {
          const stage = String(streamEvent.data.stage ?? "working");
          const message = String(streamEvent.data.message ?? "Working on your question…");
          setAgentStatus({ stage, message });
          setCompletedStages((current) =>
            current.includes(stage) ? current : [...current, stage],
          );
        }

        if (streamEvent.event === "answer_delta") {
          const delta = String(streamEvent.data.delta ?? "");
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + delta }
                : message,
            ),
          );
        }

        if (streamEvent.event === "citations" && Array.isArray(streamEvent.data.items)) {
          const citations = streamEvent.data.items as Citation[];
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId ? { ...message, citations } : message,
            ),
          );
        }

        if (streamEvent.event === "turn_completed") {
          const reviewCaseId =
            typeof streamEvent.data.review_case_id === "string"
              ? streamEvent.data.review_case_id
              : null;
          const safetyNotice = String(streamEvent.data.safety_notice ?? "");
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, reviewCaseId, safetyNotice }
                : message,
            ),
          );
        }

        if (streamEvent.event === "error") {
          throw new Error(String(streamEvent.data.message ?? "The answer could not be completed."));
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "The answer could not be completed.";
      setMessages((current) =>
        current.map((item) =>
          item.id === assistantId ? { ...item, content: item.content || message } : item,
        ),
      );
    } finally {
      setAgentStatus(null);
      setIsLoading(false);
    }
  }

  const hasConversation = messages.length > 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Pulse Care home">
          <span className="brand-mark" aria-hidden="true">+</span>
          <span>Pulse Care</span>
        </a>
        <div className="topbar-actions">
          <span className="privacy-label">Private medical assistant</span>
          <button className="new-chat-top" onClick={startNewConversation} disabled={isLoading}>
            <span aria-hidden="true">＋</span> New conversation
          </button>
          <button className="profile" onClick={changeAccessKey} title="Change access key" aria-label="Change access key">↪</button>
        </div>
      </header>

      <main className="care-stage">
        <CompanionPanel active={isLoading} status={agentStatus} />

        <section className="conversation" aria-label="Care conversation">
          <header className="conversation-header">
            <span className="ai-avatar">AI</span>
            <div><strong>Care conversation</strong><span>Sources and safety checks are always visible</span></div>
          </header>

          <div className={`message-thread ${hasConversation ? "" : "empty"}`} aria-live="polite">
            {!hasConversation && (
              <div className="welcome">
                <span className="welcome-kicker">Ask in your own words</span>
                <h1>What can I help you understand?</h1>
                <p>I can explain medical information, search your hospital’s trusted sources, and help connect you to the right department.</p>
                <div className="prompt-suggestions">
                  {["Explain my test results", "Help me prepare for a visit", "I need an appointment"].map((prompt) => (
                    <button key={prompt} onClick={() => setQuestion(prompt)}>{prompt}</button>
                  ))}
                </div>
              </div>
            )}

            {loadError && <div className="inline-error" role="alert">{loadError}</div>}

            {messages.map((message) => (
              <Message key={message.id} message={message} streaming={isLoading && !message.content} />
            ))}

            {isLoading && (
              <AgentProgress status={agentStatus} completedStages={completedStages} />
            )}
            <div ref={threadEnd} />
          </div>

          <form className="message-composer" onSubmit={submitQuestion}>
            <button className="voice-button" type="button" disabled title="Voice conversations are coming next" aria-label="Voice conversations coming soon">◉</button>
            <input value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={2000} placeholder="Ask a health question…" aria-label="Health question" disabled={isLoading} />
            <button className="send-button" type="submit" disabled={isLoading || !question.trim()} aria-label="Send question">↑</button>
          </form>
          <p className="medical-disclaimer">This assistant provides guidance, not a diagnosis. In an emergency, contact local emergency services.</p>
        </section>
      </main>
    </div>
  );
}

function CompanionPanel({ active, status }: { active: boolean; status: AgentStatus | null }) {
  return (
    <aside className="companion-panel" aria-label="Assistant status">
      <div className={`care-orb ${active ? "active" : ""}`} aria-hidden="true" />
      <div className="companion-copy">
        <span className="ready-label">{active ? "Working with care" : "Ready to help"}</span>
        <h2>Your care companion</h2>
        <p>{status?.message ?? "Listening carefully, checking trusted evidence, and connecting you to people when needed."}</p>
      </div>
    </aside>
  );
}

function AgentProgress({ status, completedStages }: { status: AgentStatus | null; completedStages: string[] }) {
  return (
    <div className="agent-progress" role="status">
      <div className="progress-dots"><i /><i /><i /></div>
      <div><strong>{status?.message ?? "Starting your care request…"}</strong><span>{completedStages.length > 0 ? `${completedStages.length} agent step${completedStages.length === 1 ? "" : "s"} completed` : "Preparing the medical agent"}</span></div>
    </div>
  );
}

function Message({ message, streaming }: { message: ChatMessage; streaming: boolean }) {
  if (message.role === "user") return <div className="user-message">{message.content}</div>;

  return (
    <article className="assistant-message">
      <div className="assistant-answer">
        {message.content ? <p>{message.content}</p> : streaming ? <span className="answer-placeholder">Preparing a careful answer…</span> : null}
      </div>

      {message.citations && message.citations.length > 0 && (
        <details className="citations">
          <summary>View {message.citations.length} source{message.citations.length === 1 ? "" : "s"}</summary>
          <div className="citation-list">
            {message.citations.map((citation) => (
              <div className="citation" key={`${citation.document_id}-${citation.chunk_id}`}>
                <span>{citation.citation_label}</span>
                <div><strong>{citation.title ?? citation.source}</strong><small>Page {citation.page_number} · {citation.source}</small></div>
              </div>
            ))}
          </div>
        </details>
      )}

      {message.safetyNotice && <div className="safety-notice"><strong>Safety note</strong><span>{message.safetyNotice}</span></div>}
      {message.reviewCaseId && <ReviewStatus caseId={message.reviewCaseId} />}
    </article>
  );
}

function ReviewStatus({ caseId }: { caseId: string }) {
  const [review, setReview] = useState<ReviewCase | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function refresh() {
      try {
        const response = await fetch(`/api/human-review/cases/${caseId}`, { cache: "no-store" });
        if (!response.ok) return;
        const nextReview = (await response.json()) as ReviewCase;
        if (cancelled) return;
        setReview(nextReview);
        if (["pending", "claimed"].includes(nextReview.status)) timer = setTimeout(refresh, 5000);
      } catch {
        if (!cancelled) timer = setTimeout(refresh, 10_000);
      }
    }

    void refresh();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, [caseId]);

  const status = review?.status ?? "pending";
  const label = status === "resolved" ? "Hospital review completed" : status === "rejected" ? "Request was not approved" : status === "expired" ? "Review request expired" : "Waiting for hospital review";

  return (
    <div className={`review-card ${status}`}>
      <span className="review-icon" aria-hidden="true">{status === "resolved" ? "✓" : "⌛"}</span>
      <div><strong>{label}</strong><p>{review?.message ?? "The responsible care team will review this request. You can keep this page open or return later."}</p></div>
    </div>
  );
}
