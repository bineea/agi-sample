import test from "node:test";
import assert from "node:assert/strict";
import {SurfaceStore, resolve, actionMessage, parseNDJSON, CATALOG_ID} from "../src/agisample/a2ui/sales_assistant/web/state.mjs";

const envelope = (kind, fields = {}) => ({version: "v0.9.1", [kind]: {surfaceId: "s", ...fields}});
function store() {
  const state = new SurfaceStore();
  state.apply(envelope("createSurface", {catalogId: CATALOG_ID}));
  return state;
}
test("数据增量更新、组件合并和删除 surface", () => {
  const state = store();
  state.apply(envelope("updateDataModel", {value: {a: {count: 1}, keep: true}}));
  state.apply(envelope("updateDataModel", {path: "/a/count", value: 2}));
  assert.equal(state.surfaces.get("s").model.a.count, 2);
  assert.equal(state.surfaces.get("s").model.keep, true);
  state.apply(envelope("updateDataModel", {path: "/a/count"}));
  assert.equal(state.surfaces.get("s").model.a.count, undefined);
  state.apply(envelope("updateComponents", {components: [{id: "root", component: "Column", children: ["x"]}]}));
  state.apply(envelope("updateComponents", {components: [{id: "x", component: "Text", text: "内容"}]}));
  assert.equal(state.surfaces.get("s").components.size, 2);
  assert.throws(() => state.apply(envelope("createSurface", {catalogId: CATALOG_ID})), /已存在/);
  state.apply(envelope("deleteSurface"));
  assert.equal(state.surfaces.size, 0);
});
test("绝对/相对路径、转义路径与点击时上下文解析", () => {
  const data = {period: "2026-09", rows: [{region: "华东"}], "a/b": {"~key": 7}};
  assert.equal(resolve({path: "/a~1b/~0key"}, data), 7);
  assert.equal(resolve({path: "region"}, data, "/rows/0"), "华东");
  const event = {name: "show_details", context: {period: {path: "/period"}, region: {path: "region"}}};
  const message = actionMessage(event, "s", "table", data, "/rows/0");
  assert.deepEqual(message.action.context, {period: "2026-09", region: "华东"});
  data.period = "2026-08";
  assert.equal(actionMessage(event, "s", "table", data, "/rows/0").action.context.period, "2026-08");
});
test("拒绝未知版本、目录、组件及危险路径", () => {
  const state = store();
  assert.throws(() => state.apply({version: "v0.8", deleteSurface: {surfaceId: "s"}}));
  assert.throws(() => state.apply(envelope("updateComponents", {components: [{id: "x", component: "Script"}]})));
  assert.throws(() => state.apply(envelope("updateDataModel", {path: "/__proto__/polluted", value: true})));
  assert.equal({}.polluted, undefined);
  assert.throws(() => state.apply({version: "v0.9.1", createSurface: {surfaceId: "other", catalogId: "unknown"}}));
});
test("NDJSON 跨字节分块、空行与无尾换行", async () => {
  const bytes = new TextEncoder().encode('{"text":"华东"}\n\n{"value":2}');
  async function* chunks() { for (const byte of bytes) yield new Uint8Array([byte]); }
  const result = [];
  for await (const message of parseNDJSON(chunks())) result.push(message);
  assert.deepEqual(result, [{text: "华东"}, {value: 2}]);
});
test("畸形 NDJSON 不被静默吞掉", async () => {
  async function* chunks() { yield new TextEncoder().encode('{"x":'); }
  await assert.rejects(async () => { for await (const _ of parseNDJSON(chunks())) {} });
});


import {renderSurface} from "../src/agisample/a2ui/sales_assistant/web/renderer.mjs";
import {execFileSync} from "node:child_process";
import {fileURLToPath} from "node:url";
import path from "node:path";

// 最小 DOM 契约用于验证节点、文本和事件；不代替真实浏览器布局验收。
class TestNode {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.events = {}; this.attributes = {}; this.text = ""; }
  append(...children) { this.children.push(...children); }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return this.text + this.children.map(child => child.textContent).join(""); }
  set innerHTML(_) { throw new Error("禁止把服务器文本作为 HTML 注入。"); }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener(name, callback) { this.events[name] = callback; }
}
function allNodes(root) { return [root, ...root.children.flatMap(allNodes)]; }
test("真实 Python 生成的消息渲染成表格，点击行回传正确区域和月份", () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const python = path.join(root, ".venv", "Scripts", "python.exe");
  const code = "import sys,json;sys.path.insert(0,'src');from agisample.a2ui.sales_assistant.service import SalesService;from datetime import date;print(json.dumps(SalesService(date(2026,9,22)).query('查看上月各区域销售情况')))";
  const messages = JSON.parse(execFileSync(python, ["-c", code], {cwd: root, encoding: "utf8"}));
  const state = new SurfaceStore();
  messages.forEach(message => state.apply(message));
  const [id, surface] = [...state.surfaces][0];
  const previous = globalThis.document;
  globalThis.document = {createElement: tag => new TestNode(tag)};
  try {
    let event;
    const rendered = renderSurface(id, surface, value => { event = value; });
    const nodes = allNodes(rendered);
    assert.equal(nodes.filter(n => n.tag === "tbody")[0].children.length, 4);
    const rowButton = nodes.find(n => n.className === "table-action");
    rowButton.events.click();
    assert.equal(event.action.name, "show_details");
    assert.equal(event.action.context.period, "2026-08");
    assert.equal(event.action.context.region, surface.model.rows[0].region);
    assert.ok(rendered.textContent.includes("2026-08"));
    const regionButton = nodes.find(n => n.dataset.componentId === "region-1");
    regionButton.events.click();
    assert.deepEqual(event.action.context, {period: "2026-08", region: "华东", view: "summary"});
  } finally { globalThis.document = previous; }
});
test("服务端字符串保持纯文本，组件循环被拒绝", () => {
  const previous = globalThis.document;
  globalThis.document = {createElement: tag => new TestNode(tag)};
  try {
    const surface = {model: {}, components: new Map([["root",
      {id: "root", component: "Text", text: '<img src=x onerror="alert(1)">'}]])};
    const node = renderSurface("s", surface, () => {});
    assert.equal(node.textContent, '<img src=x onerror="alert(1)">');
    assert.equal(node.children.length, 0);
    surface.components.set("root", {id: "root", component: "Column", children: ["root"]});
    assert.throws(() => renderSurface("s", surface, () => {}), /循环/);
  } finally { globalThis.document = previous; }
});


import {Conversation} from "../src/agisample/a2ui/sales_assistant/web/session.mjs";
test("前端在追问和点击时携带同一会话，失败保留状态，新会话清空", async () => {
  const requests = [];
  let fail = false;
  const fakeFetch = async (url, options) => {
    requests.push({url, ...options});
    if (fail) return new Response(JSON.stringify({error: "模型调用失败"}), {status: 502});
    const messages = [
      {version: "v0.9.1", createSurface: {surfaceId: "s", catalogId: CATALOG_ID}},
      {version: "v0.9.1", updateComponents: {surfaceId: "s", components: [{id: "root", component: "Text", text: "结果"}]}}
    ];
    if (requests.length > 1) messages.unshift({version: "v0.9.1", deleteSurface: {surfaceId: "s"}});
    return new Response(messages.map(m => JSON.stringify(m)).join("\n"), {headers: {"X-A2UI-Session": "session-token"}});
  };
  const conversation = new Conversation(fakeFetch);
  await conversation.turn("/api/query", {query: "本月"});
  await conversation.turn("/api/query", {query: "上月呢"});
  await conversation.turn("/api/action", {version: "v0.9.1", action: {}});
  assert.equal(requests[0].headers["X-A2UI-Session"], undefined);
  assert.equal(requests[1].headers["X-A2UI-Session"], "session-token");
  assert.equal(requests[2].headers["X-A2UI-Session"], "session-token");
  const previous = conversation.store;
  fail = true;
  await assert.rejects(() => conversation.turn("/api/query", {}), /模型调用失败/);
  assert.equal(conversation.store, previous);
  assert.equal(conversation.sessionId, "session-token");
  conversation.reset();
  assert.equal(conversation.sessionId, null);
  assert.equal(conversation.store.surfaces.size, 0);
});
