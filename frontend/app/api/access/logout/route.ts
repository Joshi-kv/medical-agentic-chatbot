import { clearSessionApiKey } from "@/lib/session";

export const runtime = "nodejs";

export async function POST() {
  await clearSessionApiKey();
  return new Response(null, { status: 204 });
}
