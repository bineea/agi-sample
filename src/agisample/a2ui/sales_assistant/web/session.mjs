// 每个页面实例独立持有会话；不把历史或会话标识存进全局共享存储。
import {SurfaceStore, parseNDJSON} from "./state.mjs";

export class Conversation {
  constructor(fetcher = fetch) {
    this.fetcher = fetcher;
    this.reset();
  }
  reset() {
    this.sessionId = null;
    this.store = new SurfaceStore();
  }
  async turn(path, payload, signal) {
    const headers = {"Content-Type": "application/json"};
    if (this.sessionId) headers["X-A2UI-Session"] = this.sessionId;
    const response = await this.fetcher(path, {method: "POST", headers, body: JSON.stringify(payload), signal});
    if (!response.ok) {
      let message = "请求失败，请检查服务后重试。";
      try { message = (await response.json()).error ?? message; } catch {}
      throw new Error(message);
    }
    const sessionId = response.headers.get("X-A2UI-Session");
    const next = new SurfaceStore();
    if (sessionId && sessionId === this.sessionId) next.surfaces = structuredClone(this.store.surfaces);
    const messages = [];
    for await (const message of parseNDJSON(response.body)) {
      next.apply(message);
      messages.push(message);
    }
    if (!messages.length) throw new Error("服务未返回界面消息，请重试。");
    this.store = next;
    this.sessionId = sessionId;
    return {store: next, messages};
  }
}
