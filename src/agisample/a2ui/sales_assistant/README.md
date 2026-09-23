# A2UI 销售数据 Agent

默认流程：

**用户提问 → 大模型理解 → 调用受限查询工具 → 模型生成 A2UI 组件 → 校验 → 前端渲染。**

连续追问和界面点击都进入同一 Agent 会话。SQL、金额计算和订单数据由确定性的 Python/SQLite 工具完成。模型决定查询参数、组件选择与布局，不再使用固定结果页面模板。

## 配置与启动

在仓库根目录 PowerShell 执行，使用现有 Python 3.11+ 虚拟环境：

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r src\agisample\a2ui\sales_assistant\requirements-agent.txt
~~~

将以下配置合并到根目录 .env，保留已有其他配置：

~~~dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=https://你的OpenAI兼容服务地址/v1
OPENAI_MODEL=你的模型名称
~~~

可参考 [.env.example](.env.example)。模型必须支持 Chat Completions 的 tools/tool_calls、JSON object 输出和 max_completion_tokens。接口地址是基础地址（通常以 /v1 结尾），不要填写完整 /chat/completions 路径。不支持这些能力的兼容接口会明确报错，不自动退回规则模式。

同时兼容项目旧变量 API_KEY、BASE_URL、MODEL。显式环境变量优先于 .env，同一来源中 OPENAI_* 优先。密钥只在后端读取，不返回前端、不写入日志。

~~~powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m agisample.a2ui.sales_assistant
~~~

浏览器打开 http://127.0.0.1:8765 。页面不会自动调用模型，点击“发送问题”后才开始请求。Ctrl+C 停止；端口冲突可加 --port 8766。

缺少密钥或模型名时启动会提示配置错误。新环境没有虚拟环境时，显式使用 py -3.13 -m venv .venv 创建。

## 体验多轮交互

1. 问“看看这个月各个区域的销售表现”。
2. 追问“只看华东”，模型根据历史继承月份。
3. 追问“那上个月呢？”，模型根据历史继承区域。
4. 若模型生成了操作按钮，点击“查看明细”；action 回到模型，由模型重新调用工具并生成界面。
5. 问“只显示销售额”，模型可选择仅生成指标卡片。
6. 问“把本月和上月并列展示”，模型可调用工具多次，生成不同表格或卡片。
7. 展开“本轮实际工具调用”和“A2UI 消息查看器”，观察工具参数、成功/拒绝结果及模型生成的组件。
8. 点击“新会话”清除当前浏览器上下文。

布局和按钮由模型决定，不保证每轮固定出现某个按钮。界面错误或模型错误不会用硬编码模板替代。旧有效结果保留，错误信息明确展示。

## 数据与计算边界

- 数据全部虚构：服务启动时以本机日期为参考，样例覆盖本月和前两个月，每月十二笔订单。
- 业务工具只支持单月份、单区域的销售汇总或订单明细；可在一轮中调用多次。
- 月份为 YYYY-MM；区域为全部、华东、华南、华北、西部；view 为 summary 或 details。
- 不提供 SQL、文件、网络浏览、写数据库或任意代码执行工具。
- SQLite 只读查询，金额按整数分保存，平均订单金额按分四舍五入。
- 模型只生成组件与简短文字；数据模型由服务器注入工具原始结果。表格数据、列名、金额、数量均通过已验证的数据绑定读取。
- 比较可并列展示工具值；未提供利润、同比/环比、差值计算工具。模型不能自行计算这些值。
- 文字解释仍由模型生成，不能将 Schema 校验视为模型语义正确性的保证。
- 查询不写入仓库已有数据库；服务跨月长期运行后需重启以更新参考日期。

## 代码分层

| 文件 | 职责 |
| --- | --- |
| llm.py | OpenAI 兼容 SDK、服务端配置、超时与脱敏错误 |
| agent.py | 会话、工具循环、历史提交/回滚 |
| agent_prompt.md | 多轮理解、查询工具及生成组件的提示词 |
| agent_tools.py | query_sales 工具定义、过滤参数白名单 |
| data.py | 虚构订单、参数化只读 SQLite 查询 |
| protocol.py / build_data | 准确金额、统计与可绑定数据；固定 build_messages 仅用于 demo |
| agent_ui.py | 官方 Schema、组件树、路径、表格列与点击来源校验 |
| server.py | HTTP JSON/NDJSON、运行模式、会话头 |
| web/session.mjs | 前端会话与响应状态提交 |
| web/app.mjs | 聊天、工具记录、错误处理 |
| web/state.mjs / renderer.mjs | A2UI 状态与白名单 DOM 渲染 |

## 模型生成什么

query_sales 工具返回 queryId 和 data。每轮工具结果保存在 /queries/q1 等路径下。模型返回如下结构（组件 ID、布局和使用的数据由模型决定）：

~~~json
{
  "reply": "已查询所选范围的销售额。",
  "components": [
    {"id": "root", "component": "Card", "child": "amount"},
    {"id": "amount", "component": "Text", "text": {"path": "/queries/q1/metrics/total"}, "variant": "metric"}
  ]
}
~~~

服务器将组件包装为 A2UI updateComponents 消息，并补充 createSurface/updateDataModel。这层确定性包装负责协议和数据完整性，不替模型选择布局。

错误 JSON、未知组件、循环/重复引用、无效绑定、模型字面量数字、错误表格列会被拒绝，并反馈给模型修正一次；仍不合法则返回错误。所有组件必须从 root 可达且每个 ID 只引用一次。本示例不支持完整官方目录。

## 协议与会话

依据 [A2UI v0.9.1](https://a2ui.org/specification/v0.9.1-a2ui/)，使用项目目录 urn:agisample:a2ui:sales:1，包含 Text、Row、Column、Card、Button、自定义 DataTable。目录定义见 [catalog.json](web/catalog.json)。

- 首轮：createSurface → updateDataModel → updateComponents。
- 后续 Agent 轮次：deleteSurface → createSurface → updateDataModel → updateComponents，清理上轮不再使用的组件。
- 客户端 action 发送 name、surfaceId、sourceComponentId、timestamp、context。服务器确认其确实来自当前会话已发布的组件，再交给 Agent。
- 会话标识使用 X-A2UI-Session 请求/响应头，不混入 A2UI envelope。每个浏览器页面实例独立保存。
- 同一会话最多十二轮，空闲三十分钟过期，服务最多六十四个会话。新会话仅清除浏览器标识，旧服务会话按过期机制清理。
- 单轮最多四次工具调用、六次模型请求，调度预算一百八十秒；单次模型请求超时最多四十五秒。同步 SDK 的网络读取行为可能使实际总时长略超预算。
- 同会话并发请求返回 409；失败不提交历史与组件。会话仅在内存，重启清空。
- 普通 HTTP POST 返回 NDJSON，不是 SSE、A2A 或逐 token 流。前端在整轮响应校验后替换界面。
- 默认仅监听本机；未提供生产环境认证/权限管理。

调用流程参考 [OpenAI Function Calling 官方文档](https://developers.openai.com/api/docs/guides/function-calling)。测试使用模拟 HTTP 接入真实 SDK，不会花费 API 额度。

## HTTP 接口

- GET /api/info：返回 mode、model、referenceDate，不返回密钥或接口认证配置。
- POST /api/query：JSON {"query":"那上个月呢？"}；首次无会话头，后续带 X-A2UI-Session。
- POST /api/action：标准 A2UI action JSON，同样带 X-A2UI-Session。
- 错误返回中文 {"error":"原因"}。400 无效请求、409 会话失效/忙、413 过大请求、415 内容类型错误、429 容量限制、502 模型/生成失败、504 超时。

## 显式规则 demo

只用于无密钥学习和回归验证，没有大模型或多轮理解：

~~~powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m agisample.a2ui.sales_assistant --mode demo
~~~

此模式仍只需 Python 标准库。默认 agent 不会自动降级到它。

## 验证

安装 requirements-agent.txt 后：

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_a2ui_*.py" -v
node --test tests/a2ui_renderer.test.mjs
~~~

Node 用例需要 Node 22+，无需 npm install；跨层用例使用项目 Windows 虚拟环境。

已覆盖 SDK 模拟 HTTP 的真实请求格式、工具结果回传、模型自选布局、多轮/点击、独立会话、拒绝非法工具与 SQL 参数、生成失败回滚、Schema、已知金额、协议解析、DOM 事件与前端会话传播。

**验收边界：** 真实兼容服务的地址、模型名及密钥尚未配置，因此真实模型端到端效果未验收；浏览器视觉与实际点击未验收。离线模拟通过不代表远端模型兼容性已通过。

本次本地验证结果：27 项 Python 测试和 8 项 Node 测试通过；Python 编译及前端模块语法检查通过。
