import { backendFetch, proxyJson, unavailableResponse } from "@/lib/backend";
import { setSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { apiKey?: unknown } | null;
  const apiKey = typeof body?.apiKey === "string" ? body.apiKey.trim() : "";

  if (!apiKey || apiKey.length > 256) {
    return Response.json(
      { detail: { code: "invalid_api_key", message: "Enter a valid API key." } },
      { status: 400 },
    );
  }

  try {
    const response = await backendFetch("/v1/auth/me", {}, apiKey);
    if (!response.ok) return proxyJson(response);

    const identity = await response.json();
    await setSessionApiKey(apiKey);
    return Response.json(identity, { headers: { "Cache-Control": "no-store" } });
  } catch {
    return unavailableResponse();
  }
}
