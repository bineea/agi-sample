import {Conversation} from "./session.mjs";
import {renderSurface} from "./renderer.mjs";

const form = document.querySelector("#query-form");
const input = document.querySelector("#query");
const results = document.querySelector("#results");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const trace = document.querySelector("#trace");
const traceCount = document.querySelector("#trace-count");
const history = document.querySelector("#conversation");
const toolLog = document.querySelector("#tool-log");
const conversation = new Conversation();
let busy = false, mode = "agent";

function setBusy(value) {
  busy = value;
  document.querySelectorAll("button, textarea").forEach(control => { control.disabled = value; });
  results.setAttribute("aria-busy", String(value));
  status.textContent = value ? (mode === "agent" ? "Agent 正在处理…" : "正在查询…") : "等待提问";
}
function addMessage(role, text) {
  const message = document.createElement("p");
  message.className = role === "用户" ? "user-message" : "assistant-message";
  const label = document.createElement("strong");
  label.textContent = role + "：";
  message.append(label, document.createTextNode(text));
  history.append(message);
  history.scrollTop = history.scrollHeight;
}
function actionLabel(action) {
  const names = {show_details: "查看明细", show_summary: "查看汇总", filter_region: "筛选区域"};
  return "点击“" + (names[action.name] ?? action.name) + "”：" + Object.values(action.context).join(" / ");
}
async function request(path, payload, userText) {
  if (busy) return;
  setBusy(true);
  error.hidden = true;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 200000);
  trace.textContent = "发送\n" + JSON.stringify(payload, null, 2);
  traceCount.textContent = "等待响应";
  try {
    const {store, messages} = await conversation.turn(path, payload, controller.signal);
    const fragment = document.createDocumentFragment();
    let agent = null, demoTitle = "";
    for (const [id, surface] of store.surfaces) {
      fragment.append(renderSurface(id, surface, message => request("/api/action", message, actionLabel(message.action))));
      agent = surface.model.agent ?? null;
      demoTitle = surface.model.title ?? "";
    }
    results.replaceChildren(fragment);
    results.classList.remove("initial");
    addMessage("用户", userText);
    addMessage("助手", agent?.reply ?? ("规则演示已完成：" + demoTitle));
    toolLog.replaceChildren();
    for (const step of agent?.steps ?? []) {
      const row = document.createElement("li");
      row.textContent = step.tool + " · " + step.status + (step.arguments ? " · " + JSON.stringify(step.arguments) : " · " + step.error);
      toolLog.append(row);
    }
    document.querySelector("#tool-panel").hidden = !agent;
    if (agent && !agent.steps.length) {
      const row = document.createElement("li");
      row.textContent = "本轮未调用查询工具（例如解释或澄清问题）。";
      toolLog.append(row);
    }
    trace.textContent += "\n\n接收\n" + messages.map(m => JSON.stringify(m, null, 2)).join("\n\n");
    traceCount.textContent = messages.length + " 条消息";
    if (path === "/api/query") input.value = "";
    input.placeholder = mode === "agent" ? "继续追问，例如：那上个月呢？" : "输入完整查询条件";
  } catch (exc) {
    error.textContent = exc.name === "AbortError" ? "处理超时，服务端可能仍在完成本轮，请稍后重试。" :
      exc instanceof TypeError ? "无法连接服务或渲染响应，请确认后端状态。" : exc.message;
    error.hidden = false;
    trace.textContent += "\n\n错误\n" + error.textContent;
    traceCount.textContent = "请求失败";
  } finally {
    clearTimeout(timer);
    setBusy(false);
    status.textContent = error.hidden ? "本轮已完成" : "本轮未完成";
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  request("/api/query", {query: input.value}, input.value);
});
document.querySelectorAll("[data-query]").forEach(button => {
  button.addEventListener("click", () => {
    input.value = button.dataset.query;
    form.requestSubmit();
  });
});
document.querySelector("#new-chat").addEventListener("click", () => {
  conversation.reset();
  history.replaceChildren();
  toolLog.replaceChildren();
  results.replaceChildren();
  const hint = document.createElement("p");
  hint.textContent = "已开启新会话。请提出新的数据问题。";
  results.append(hint);
  document.querySelector("#tool-panel").hidden = true;
  trace.textContent = "";
  traceCount.textContent = "0 条消息";
  error.hidden = true;
  input.value = "";
  input.placeholder = "例如：查看本月各区域销售情况";
  status.textContent = "等待提问";
  input.focus();
});

setBusy(true);
try {
  const response = await fetch("/api/info");
  if (!response.ok) throw new Error("无法获取服务配置。");
  const info = await response.json();
  mode = info.mode;
  document.querySelector("#mode-label").textContent = mode === "agent" ? "Agent · " + info.model + " · 虚构数据" : "规则 demo · 不调用模型";
  document.querySelector("#scope-note").textContent = mode === "agent"
    ? "模型理解需求、调用只读工具并生成界面。支持连续追问和界面点击；订单为虚构数据，模型调用会使用你配置的接口。"
    : "当前为显式规则 demo。没有大模型与多轮理解；输入完整月份、区域和视图。";
  input.maxLength = mode === "agent" ? 2000 : 500;
  setBusy(false);
} catch (exc) {
  error.textContent = exc.message;
  error.hidden = false;
  status.textContent = "服务未就绪，请刷新页面重试";
}
