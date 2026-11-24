import { backendFetch, proxyJson, unavailableResponse } from "@/lib/backend";
import { setSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";

type CreatedKey = { api_key?: unknown };

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { name?: unknown } | null;
  const name = typeof body?.name === "string" ? body.name.trim() : "";

  if (name.length < 2 || name.length > 100) {
    return Response.json(
      { detail: { code: "invalid_name", message: "Name must be between 2 and 100 characters." } },
      { status: 400 },
    );
  }

  try {
    const response = await backendFetch("/v1/auth/api-keys", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    if (!response.ok) return proxyJson(response);

    const created = (await response.json()) as CreatedKey;
    if (typeof created.api_key !== "string") return unavailableResponse();

    await setSessionApiKey(created.api_key);
    return Response.json(created, { status: 201, headers: { "Cache-Control": "no-store" } });
  } catch {
    return unavailableResponse();
  }
}
