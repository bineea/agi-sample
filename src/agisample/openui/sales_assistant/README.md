# OpenUI 销售数据 Agent

默认流程：

**用户提问 → 大模型流式生成 openui-lang → 前端 `<Renderer>` 渐进渲染为 Vue 组件 → 界面中的 Query 语句由前端 toolProvider 回调后端只读工具取数。**

连续追问和界面按钮点击（@ToAssistant）都进入同一 Agent 会话。SQL 和金额计算由确定性的 Python/SQLite 工具完成；模型只决定查询参数、组件选择与布局。数据取数不经过模型：模型写 `Query("query_sales", {period, region, view}, 默认值)`，前端在渲染时调用后端执行，界面上的筛选器（$variable 绑定）变化时前端自动重新查询，无需再请求模型。

与 [a2ui/sales_assistant](../../a2ui/sales_assistant/README.md) 同业务、不同界面协议：A2UI 版由服务端校验组件 JSON 并注入数据模型；本版使用 [OpenUI Lang](https://www.openui.com/docs/openui-lang) 0.5 与官方 Vue 运行时 [@openuidev/vue-lang](https://github.com/thesysdev/openui/tree/main/packages/vue-lang)，组件库为自定义 Vue 组件（Element Plus + ECharts 封装）。

## 配置与启动

在仓库根目录 PowerShell 执行，使用现有 Python 3.11+ 虚拟环境：

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r src\agisample\openui\sales_assistant\requirements-agent.txt
~~~

将以下配置合并到根目录 .env（可参考 [.env.example](.env.example)），保留已有其他配置：

~~~dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://你的OpenAI兼容服务地址/v1
OPENAI_MODEL=你的模型名称
~~~

模型需支持 Chat Completions 流式输出。同时兼容项目旧变量 API_KEY、BASE_URL、MODEL。密钥只在后端读取。

启动后端（端口 8766）：

~~~powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m agisample.openui.sales_assistant
~~~

启动前端（开发模式，端口 5174，/api 自动代理到后端）：

~~~powershell
cd src\agisample\openui\sales_assistant\web
npm install
npm run dev
~~~

浏览器打开 http://localhost:5174 。生产模式可 `npm run build` 后只启动后端，由后端托管 web/dist。

## 离线演示模式

无密钥时学习链路用（预置两段 openui-lang 响应，Query 取数仍为真实后端 + SQLite）：

~~~powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m agisample.openui.sales_assistant --mode demo
~~~

默认 agent 模式不会自动降级到 demo。

## 体验多轮交互

1. 问“看看这个月各个区域的销售表现”，模型流式生成仪表盘（StatCard + 区域柱状图）。
2. 在生成的界面上切换“区域”下拉框：$variable 绑定变化，前端自动重新执行 Query，不调用模型。
3. 点击模型生成的“查看订单明细”按钮：@ToAssistant 动作把消息（含表单状态）作为下一轮输入发给模型。
4. 追问“那上个月呢？”，模型根据历史继承条件。
5. 展开每轮下方的“本轮 openui-lang 源码”，查看模型实际输出的语句与解析状态。
6. 点击“新会话”清除当前会话。

布局和按钮由模型决定，不保证每轮固定出现某个组件。

## 数据与计算边界

- 数据全部虚构：以本机日期为参考，覆盖本月和前两个月，每月每区域三笔订单。
- 工具只支持单月份、单区域的 summary/details；金额按整数分存储，服务端格式化为 ¥ 字符串，图表用数值字段 total_value（元）。
- 不提供 SQL、文件、网络、写库或任意代码执行工具；模型看不到数值，只能通过绑定引用查询结果字段。
- 会话：X-OpenUI-Session 请求/响应头；每会话最多十二轮，空闲三十分钟过期，最多六十四个会话；仅内存保存，重启清空。
- 单轮生成预算一百八十秒，单次模型请求最长九十秒；同会话并发请求返回 409；生成失败不提交历史。

## HTTP 接口

- GET /api/info：返回 mode、model、referenceDate。
- POST /api/chat：JSON {"query":"那上个月呢？"}；响应为 NDJSON 流（delta/done/error 事件），首行响应头携带 X-OpenUI-Session。
- POST /api/tools/query_sales：JSON {"arguments":{"period":"2026-10","region":"全部","view":"summary"}}；供前端 toolProvider 执行 Query 语句，不需要会话。
- 错误返回中文 {"error":"原因"}。400 无效请求、409 会话失效/忙、413 过大请求、415 内容类型错误、429 容量限制、502 模型失败、504 超时。

## 代码分层

| 文件 | 职责 |
| --- | --- |
| llm.py | OpenAI 兼容 SDK 流式输出、服务端配置、脱敏错误 |
| agent.py | 会话管理、流式轮次、历史提交/回滚 |
| agent_prompt.md | 业务规则与输出约束提示词 |
| library_prompt.txt | 由组件库 schema 自动生成的 OpenUI Lang 提示词（勿手改） |
| agent_tools.py | query_sales 参数白名单与结果组装 |
| data.py | 虚构订单、参数化只读 SQLite 查询 |
| demo.py | 离线预置响应模型（--mode demo） |
| server.py | HTTP：NDJSON 流式对话、工具端点、静态托管 |
| web/src/openui/library/spec.ts | 组件库 zod schema 单一事实来源 |
| web/scripts/generate-prompt.ts | 从 spec.ts 生成 library_prompt.txt |
| web/src/openui/library/index.ts | Vue 组件库注册（schema + 视图绑定） |
| web/src/openui/components/*.vue | Element Plus / ECharts 组件渲染实现 |
| web/src/App.vue | 对话外壳：消息流、流式渲染、动作回话 |
| web/src/api.ts | NDJSON 流客户端、工具转发、会话头管理 |

## 组件库与提示词同步

模型的“词汇表”来自 `web/src/openui/library/spec.ts` 的 zod schema。修改组件后必须重新生成提示词：

~~~powershell
cd src\agisample\openui\sales_assistant\web
npm run generate-prompt
~~~

## 验证

本次本地验证（demo 模式，除模型厂商外全链路）：流式对话、仪表盘渲染、区域切换自动重新查询、@ToAssistant 按钮回话、订单明细表格（12 行真实数据）、无控制台错误。

**验收边界：** 真实兼容服务的地址、模型名及密钥尚未配置，真实模型的多轮理解与界面生成质量未验收；demo 模式通过不代表远端模型兼容性已通过。
