/** 后端 API 客户端：NDJSON 流式对话 + 业务工具转发 + 会话头管理 */

const SESSION_HEADER = "X-OpenUI-Session";

let sessionId: string | null = null;

export function resetSession() {
  sessionId = null;
}

export interface ChatEvent {
  type: "delta" | "done" | "error";
  text?: string;
  error?: string;
  status?: number;
  turns?: number;
}

export async function* chatStream(query: string): AsyncGenerator<ChatEvent> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(sessionId ? { [SESSION_HEADER]: sessionId } : {}),
    },
    body: JSON.stringify({ query }),
  });
  const sid = res.headers.get(SESSION_HEADER);
  if (sid) sessionId = sid;
  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => null);
    yield { type: "error", error: err?.error ?? `请求失败（HTTP ${res.status}）`, status: res.status };
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (line) yield JSON.parse(line) as ChatEvent;
    }
  }
}

export async function callTool(name: string, args: Record<string, unknown>) {
  const res = await fetch(`/api/tools/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ arguments: args }),
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.error ?? `工具调用失败（HTTP ${res.status}）`);
  return data;
}

export interface ServiceInfo {
  mode: string;
  model: string | null;
  referenceDate: string;
}

export async function fetchInfo(): Promise<ServiceInfo | null> {
  try {
    const res = await fetch("/api/info");
    return res.ok ? await res.json() : null;
  } catch {
    return null;
  }
}
