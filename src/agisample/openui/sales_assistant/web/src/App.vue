<script setup lang="ts">
import { nextTick, onMounted, ref } from "vue";
import { ElMessage } from "element-plus";
import { Renderer, provideFormName } from "@openuidev/vue-lang";
import type { ActionEvent, OpenUIError, ParseResult } from "@openuidev/vue-lang";
import { demoLibrary } from "./openui/library";
import { callTool, chatStream, fetchInfo, resetSession, type ServiceInfo } from "./api";

// 按钮可能不在 Form 内（如全局追问按钮），提供空的根表单名消除 inject 警告
provideFormName(ref(undefined));

// ── 消息列表 ───────────────────────────────────────────

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  streaming: boolean;
  error?: string;
  parseErrors?: OpenUIError[];
  statements?: number;
}

let nextId = 1;
const messages = ref<ChatMessage[]>([]);
const sending = ref(false);
const input = ref("");
const info = ref<ServiceInfo | null>(null);
const listEl = ref<HTMLDivElement>();

const STARTERS = [
  "看看这个月各个区域的销售表现",
  "本月华东的订单明细",
  "把本月和上月并列对比一下",
];

onMounted(async () => {
  info.value = await fetchInfo();
});

async function scrollToBottom() {
  await nextTick();
  listEl.value?.scrollTo({ top: listEl.value.scrollHeight, behavior: "smooth" });
}

// ── 发送与流式接收 ─────────────────────────────────────

async function send(text: string) {
  const query = text.trim();
  if (!query || sending.value) return;
  sending.value = true;
  input.value = "";
  messages.value.push({ id: nextId++, role: "user", content: query, streaming: false });
  const answer: ChatMessage = { id: nextId++, role: "assistant", content: "", streaming: true };
  messages.value.push(answer);
  await scrollToBottom();
  try {
    for await (const event of chatStream(query)) {
      if (event.type === "delta") {
        answer.content += event.text ?? "";
        scrollToBottom();
      } else if (event.type === "error") {
        answer.error = event.error;
      }
    }
  } catch {
    answer.error = "网络异常，请稍后重试。";
  } finally {
    answer.streaming = false;
    sending.value = false;
    scrollToBottom();
  }
}

function newSession() {
  resetSession();
  messages.value = [];
  ElMessage.success("已开始新会话。");
}

// ── 组件动作：@ToAssistant 回话成下一轮用户消息 ────────

function handleAction(event: ActionEvent) {
  if (event.type === "continue_conversation") {
    const formState = event.formState ?? {};
    const hasFormData = Object.keys(formState).length > 0;
    const text = event.humanFriendlyMessage +
      (hasFormData ? `\n表单提交内容：${JSON.stringify(formState, ensureUtf8Safe(formState))}` : "");
    send(text);
  } else if (event.type === "open_url") {
    const url = (event.params as any)?.url;
    if (typeof url === "string" && /^https?:\/\//.test(url)) window.open(url, "_blank", "noreferrer");
  }
}

// formState 中可能包含 undefined 等非 JSON 值，先序列化一遍再展示
function ensureUtf8Safe(value: unknown) {
  return JSON.parse(JSON.stringify(value));
}

// ── toolProvider：Query()/Mutation() 转发到 Python 后端 ─

const toolProvider = {
  query_sales: (args: Record<string, unknown>) => callTool("query_sales", args),
};

function onParseErrors(msg: ChatMessage, errors: OpenUIError[]) {
  msg.parseErrors = errors;
}

function onParseResult(msg: ChatMessage, result: ParseResult | null) {
  msg.statements = result?.meta?.statementCount ?? 0;
}
</script>

<template>
  <div class="page">
    <header class="topbar">
      <div>
        <h1>OpenUI 销售查询助手</h1>
        <p class="subtitle">
          模型实时生成 openui-lang，由 <code>@openuidev/vue-lang</code> 渲染为 Vue 组件
        </p>
      </div>
      <div class="topbar-right">
        <el-tag v-if="info?.model" size="small" effect="plain">{{ info.model }}</el-tag>
        <el-tag v-else size="small" type="danger" effect="plain">后端未连接</el-tag>
        <el-button size="small" @click="newSession">新会话</el-button>
      </div>
    </header>

    <main ref="listEl" class="message-list">
      <div v-if="!messages.length" class="welcome">
        <h2>试试这些问题</h2>
        <div class="starters">
          <el-card
            v-for="s in STARTERS"
            :key="s"
            shadow="hover"
            class="starter"
            @click="send(s)"
          >
            {{ s }}
          </el-card>
        </div>
        <p class="tip">
          追问“只看华东”“那上个月呢”可继承上下文；界面中的筛选、按钮由模型生成并可直接交互。
        </p>
      </div>

      <template v-for="msg in messages" :key="msg.id">
        <div v-if="msg.role === 'user'" class="row user">
          <div class="bubble user-bubble">{{ msg.content }}</div>
        </div>
        <div v-else class="row assistant">
          <div class="assistant-body">
            <el-alert
              v-if="msg.error"
              :title="msg.error"
              type="error"
              :closable="false"
              show-icon
            />
            <Renderer
              v-if="msg.content"
              :response="msg.content"
              :library="demoLibrary"
              :is-streaming="msg.streaming"
              :tool-provider="toolProvider"
              :on-action="handleAction"
              :on-error="(errs: OpenUIError[]) => onParseErrors(msg, errs)"
              :on-parse-result="(r: ParseResult | null) => onParseResult(msg, r)"
            />
            <div v-if="msg.streaming" class="streaming-hint">
              <span class="dot"></span>正在生成界面…
            </div>
            <el-collapse v-if="!msg.streaming && msg.content" class="debug">
              <el-collapse-item
                :title="`本轮 openui-lang 源码（${msg.statements ?? 0} 条语句${msg.parseErrors?.length ? '，' + msg.parseErrors.length + ' 个解析错误' : ''}）`"
              >
                <pre class="source">{{ msg.content }}</pre>
                <div v-for="(e, i) in msg.parseErrors ?? []" :key="i" class="parse-error">
                  <el-tag size="small" type="danger">{{ e.code }}</el-tag> {{ e.message }}
                </div>
              </el-collapse-item>
            </el-collapse>
          </div>
        </div>
      </template>
    </main>

    <footer class="composer">
      <el-input
        v-model="input"
        type="textarea"
        :autosize="{ minRows: 1, maxRows: 4 }"
        placeholder="询问销售数据，例如：看看这个月各个区域的销售表现"
        @keydown.enter.exact.prevent="send(input)"
      />
      <el-button type="primary" :loading="sending" @click="send(input)">发送</el-button>
    </footer>
  </div>
</template>

<style scoped>
.page {
  max-width: 960px;
  height: 100vh;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  padding: 0 16px;
}
.topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0 12px;
  border-bottom: 1px solid #e5e6eb;
}
.topbar h1 {
  margin: 0;
  font-size: 18px;
}
.subtitle {
  margin: 4px 0 0;
  font-size: 12px;
  color: #86909c;
}
.topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 16px 0;
}
.welcome {
  text-align: center;
  padding-top: 48px;
}
.welcome h2 {
  font-size: 16px;
  color: #4e5969;
}
.starters {
  display: flex;
  gap: 12px;
  justify-content: center;
  flex-wrap: wrap;
}
.starter {
  cursor: pointer;
  width: 220px;
  font-size: 14px;
}
.tip {
  margin-top: 16px;
  font-size: 12px;
  color: #86909c;
}
.row {
  display: flex;
  margin-bottom: 16px;
}
.row.user {
  justify-content: flex-end;
}
.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
}
.user-bubble {
  background: #165dff;
  color: #fff;
}
.assistant-body {
  width: 100%;
}
.streaming-hint {
  margin-top: 8px;
  font-size: 12px;
  color: #86909c;
  display: flex;
  align-items: center;
  gap: 6px;
}
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #165dff;
  animation: pulse 1s infinite;
}
@keyframes pulse {
  50% {
    opacity: 0.2;
  }
}
.debug {
  margin-top: 10px;
  --el-collapse-header-font-size: 12px;
}
.source {
  background: #1d2129;
  color: #a9dc76;
  border-radius: 8px;
  padding: 12px;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 320px;
  overflow: auto;
}
.parse-error {
  font-size: 12px;
  margin-top: 6px;
}
.composer {
  display: flex;
  gap: 10px;
  padding: 12px 0 16px;
  border-top: 1px solid #e5e6eb;
  align-items: flex-end;
}
</style>
