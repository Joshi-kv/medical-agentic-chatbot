import { ChatShell } from "@/components/chat-shell";
import { getSessionApiKey } from "@/lib/session";
import { redirect } from "next/navigation";

export default async function HomePage() {
  if (!(await getSessionApiKey())) redirect("/access");
  return <ChatShell />;
}
