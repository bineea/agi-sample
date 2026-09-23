// 教学用 A2UI v0.9.1 状态层；仅支持本示例目录，不执行远端代码。
export const CATALOG_ID = "urn:agisample:a2ui:sales:1";
const TYPES = new Set(["Text", "Column", "Row", "Card", "Button", "DataTable"]);
const KINDS = ["createSurface", "updateComponents", "updateDataModel", "deleteSurface"];
const forbidden = new Set(["__proto__", "prototype", "constructor"]);

function segments(path) {
  if (typeof path !== "string") throw new Error("数据路径必须是字符串。");
  if (!path || path === "/") return [];
  const parts = path.replace(/^\//, "").split("/").map(p => p.replace(/~1/g, "/").replace(/~0/g, "~"));
  if (parts.some(p => forbidden.has(p))) throw new Error("拒绝不安全的数据路径。");
  return parts;
}

export function resolve(value, model, scope = "") {
  if (!value || typeof value !== "object" || !Object.hasOwn(value, "path")) return value;
  const path = value.path.startsWith("/") ? value.path : scope + "/" + value.path;
  return segments(path).reduce((node, key) =>
    node !== null && typeof node === "object" && Object.hasOwn(node, key) ? node[key] : undefined, model);
}

function updatePath(model, path, value, remove) {
  const parts = segments(path);
  if (!parts.length) return remove ? {} : structuredClone(value);
  if (model === null || typeof model !== "object") throw new Error("无法在非对象数据模型中写入路径。");
  let node = model;
  for (let i = 0; i < parts.length - 1; i++) {
    const key = parts[i];
    if (Array.isArray(node) && !/^(0|[1-9]\d*)$/.test(key)) throw new Error("数组路径必须是数字索引。");
    if (!Object.hasOwn(node, key)) node[key] = {};
    node = node[key];
    if (node === null || typeof node !== "object") throw new Error("数据路径中间节点不是对象。");
  }
  const key = parts.at(-1);
  if (Array.isArray(node) && !/^(0|[1-9]\d*)$/.test(key)) throw new Error("数组路径必须是数字索引。");
  if (remove) {
    if (Array.isArray(node)) node.splice(Number(key), 1);
    else delete node[key];
  } else node[key] = structuredClone(value);
  return model;
}

export class SurfaceStore {
  constructor() { this.surfaces = new Map(); }
  apply(message) {
    if (!message || message.version !== "v0.9.1") throw new Error("不支持的 A2UI 协议版本。");
    const kinds = KINDS.filter(key => Object.hasOwn(message, key));
    if (kinds.length !== 1) throw new Error("每条消息必须包含一种操作。");
    const kind = kinds[0], payload = message[kind];
    if (!payload || typeof payload.surfaceId !== "string") throw new Error("缺少 surfaceId。");
    const id = payload.surfaceId;
    if (kind === "createSurface") {
      if (this.surfaces.has(id)) throw new Error("界面已存在，需先删除。");
      if (payload.catalogId !== CATALOG_ID) throw new Error("不支持此组件目录。");
      this.surfaces.set(id, {components: new Map(), model: {}});
      return;
    }
    if (kind === "deleteSurface") { this.surfaces.delete(id); return; }
    const surface = this.surfaces.get(id);
    if (!surface) throw new Error("请先创建界面。");
    if (kind === "updateDataModel") {
      surface.model = updatePath(surface.model, payload.path ?? "/", payload.value, !Object.hasOwn(payload, "value"));
    } else {
      if (!Array.isArray(payload.components)) throw new Error("组件列表格式错误。");
      for (const c of payload.components) {
        if (!c || typeof c.id !== "string" || !TYPES.has(c.component)) throw new Error("不支持的组件。");
        if (["Row", "Column"].includes(c.component) && !Array.isArray(c.children))
          throw new Error("示例仅支持显式子组件列表。");
        surface.components.set(c.id, structuredClone(c));
      }
    }
  }
}

export function actionMessage(event, surfaceId, sourceComponentId, model, scope = "") {
  const context = {};
  for (const [key, value] of Object.entries(event.context ?? {})) {
    if (forbidden.has(key)) throw new Error("不安全的事件字段。");
    context[key] = resolve(value, model, scope);
  }
  return {version: "v0.9.1", action: {name: event.name, surfaceId, sourceComponentId,
    timestamp: new Date().toISOString(), context}};
}

export async function* parseNDJSON(chunks) {
  const decoder = new TextDecoder("utf-8", {fatal: true});
  let pending = "";
  for await (const chunk of chunks) {
    pending += decoder.decode(chunk, {stream: true});
    if (pending.length > 2_000_000) throw new Error("协议消息过大。");
    let newline;
    while ((newline = pending.indexOf("\n")) >= 0) {
      const line = pending.slice(0, newline).trim();
      pending = pending.slice(newline + 1);
      if (line) yield JSON.parse(line);
    }
  }
  pending += decoder.decode();
  if (pending.trim()) yield JSON.parse(pending);
}
