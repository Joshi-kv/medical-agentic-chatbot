import { backendFetch, proxyJson, unavailableResponse, unauthorizedResponse } from "@/lib/backend";
import { getSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const apiKey = await getSessionApiKey();
  if (!apiKey) return unauthorizedResponse();
  try {
    return proxyJson(
      await backendFetch("/v1/conversations", {
        method: "POST",
        body: await request.text(),
      }, apiKey),
    );
  } catch {
    return unavailableResponse();
  }
}
