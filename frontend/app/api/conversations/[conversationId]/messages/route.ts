import { backendFetch, proxyJson, unavailableResponse, unauthorizedResponse } from "@/lib/backend";
import { getSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";

type RouteContext = { params: Promise<{ conversationId: string }> };

export async function GET(_request: Request, context: RouteContext) {
  const apiKey = await getSessionApiKey();
  if (!apiKey) return unauthorizedResponse();
  const { conversationId } = await context.params;
  try {
    return proxyJson(
      await backendFetch(
        `/v1/conversations/${encodeURIComponent(conversationId)}/messages`,
        {},
        apiKey,
      ),
    );
  } catch {
    return unavailableResponse();
  }
}
