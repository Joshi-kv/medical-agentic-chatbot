import { backendFetch, proxyJson, unavailableResponse, unauthorizedResponse } from "@/lib/backend";
import { getSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ conversationId: string }> };

export async function POST(request: Request, context: RouteContext) {
  const apiKey = await getSessionApiKey();
  if (!apiKey) return unauthorizedResponse();
  const { conversationId } = await context.params;

  try {
    const response = await backendFetch(
      `/v1/conversations/${encodeURIComponent(conversationId)}/turns/stream`,
      { method: "POST", body: await request.text(), signal: request.signal },
      apiKey,
    );

    if (!response.ok || !response.body) return proxyJson(response);

    return new Response(response.body, {
      status: response.status,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch {
    return unavailableResponse();
  }
}
