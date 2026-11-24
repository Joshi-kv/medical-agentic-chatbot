"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";

type Mode = "existing" | "new";
type CreatedKey = { api_key: string; key_prefix: string; name: string };

function errorMessage(payload: unknown): string {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) {
    return "The request could not be completed. Please try again.";
  }
  const detail = payload.detail;
  if (detail && typeof detail === "object" && "message" in detail && typeof detail.message === "string") {
    return detail.message;
  }
  return "The request could not be completed. Please try again.";
}

export function AccessPanel() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("existing");
  const [apiKey, setApiKey] = useState("");
  const [name, setName] = useState("");
  const [created, setCreated] = useState<CreatedKey | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submitExisting(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/access/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ apiKey }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(errorMessage(payload));
      router.replace("/");
      router.refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The key could not be verified.");
    } finally {
      setLoading(false);
    }
  }

  async function generateKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/access/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(errorMessage(payload));
      setCreated(payload as CreatedKey);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The key could not be created.");
    } finally {
      setLoading(false);
    }
  }

  async function copyCreatedKey() {
    if (!created) return;
    await navigator.clipboard.writeText(created.api_key);
    setCopied(true);
  }

  function chooseMode(nextMode: Mode) {
    setMode(nextMode);
    setError(null);
  }

  return (
    <main className="access-page">
      <section className="access-intro">
        <a className="brand access-brand" href="/" aria-label="Pulse Care home">
          <span className="brand-mark" aria-hidden="true">+</span>
          <span>Pulse Care</span>
        </a>
        <div className="access-orb" aria-hidden="true" />
        <p className="access-kicker">Your private care space</p>
        <h1>Start a secure conversation.</h1>
        <p className="access-description">
          Use an existing access key, or create a personal key for this browser. Your key stays in a protected, HttpOnly session.
        </p>
        <div className="access-trust">
          <span>Private session</span>
          <span>Evidence-backed answers</span>
          <span>Human review when needed</span>
        </div>
      </section>

      <section className="access-card" aria-labelledby="access-heading">
        <div>
          <span className="access-step">Secure access</span>
          <h2 id="access-heading">How would you like to continue?</h2>
          <p>Your name labels a key; it does not verify your identity.</p>
        </div>

        <div className="access-tabs" role="tablist" aria-label="Access method">
          <button className={mode === "existing" ? "active" : ""} onClick={() => chooseMode("existing")} role="tab" aria-selected={mode === "existing"}>I have a key</button>
          <button className={mode === "new" ? "active" : ""} onClick={() => chooseMode("new")} role="tab" aria-selected={mode === "new"}>Create a key</button>
        </div>

        {mode === "existing" && (
          <form className="access-form" onSubmit={submitExisting}>
            <label htmlFor="api-key">API key</label>
            <input id="api-key" type="password" autoComplete="off" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="med_••••••••••••••••" required />
            <p className="field-help">The key is verified by the medical API and is never exposed to browser scripts after sign-in.</p>
            {error && <p className="access-error" role="alert">{error}</p>}
            <button className="access-primary" disabled={loading || !apiKey.trim()}>{loading ? "Verifying…" : "Continue to Pulse Care"}</button>
          </form>
        )}

        {mode === "new" && !created && (
          <form className="access-form" onSubmit={generateKey}>
            <label htmlFor="display-name">Your name</label>
            <input id="display-name" autoComplete="name" value={name} onChange={(event) => setName(event.target.value)} placeholder="For example, Maya" minLength={2} maxLength={100} required />
            <p className="field-help">We use this only as the key label. Keep the generated key somewhere safe.</p>
            {error && <p className="access-error" role="alert">{error}</p>}
            <button className="access-primary" disabled={loading || name.trim().length < 2}>{loading ? "Creating your key…" : "Generate secure key"}</button>
          </form>
        )}

        {mode === "new" && created && (
          <div className="created-key" aria-live="polite">
            <span className="created-check" aria-hidden="true">✓</span>
            <h3>Your key is ready, {created.name}.</h3>
            <p>This is the only time the complete key will be shown. Copy it before continuing.</p>
            <div className="key-value"><code>{created.api_key}</code><button onClick={copyCreatedKey}>{copied ? "Copied" : "Copy"}</button></div>
            <button className="access-primary" onClick={() => router.replace("/")}>I saved it — continue</button>
          </div>
        )}

        <p className="access-disclaimer">Pulse Care provides medical guidance, not a diagnosis. For emergencies, contact local emergency services.</p>
      </section>
    </main>
  );
}
