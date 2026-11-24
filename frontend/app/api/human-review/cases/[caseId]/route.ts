import { backendFetch, proxyJson, unavailableResponse, unauthorizedResponse } from "@/lib/backend";
import { getSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ caseId: string }> };

export async function GET(_request: Request, context: RouteContext) {
  const apiKey = await getSessionApiKey();
  if (!apiKey) return unauthorizedResponse();
  const { caseId } = await context.params;
  try {
    return proxyJson(
      await backendFetch(
        `/v1/human-review/cases/${encodeURIComponent(caseId)}`,
        {},
        apiKey,
      ),
    );
  } catch {
    return unavailableResponse();
  }
}
