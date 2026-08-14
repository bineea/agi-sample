# agi-sample

`agi-sample` 是一个用于尝试 Python AI/LLM 新技术、新组件和新框架能力的 demo 仓库。它不是生产 SDK，而是用于快速验证 LangChain、LangGraph、AgentScope、RAG、文档智能、向量检索、本地模型和传统机器学习匹配等能力的样例集合。

## 项目定位

本项目适合用于：

- 快速验证新的 LLM 框架、Agent 框架和工具调用模式；
- 沉淀 RAG、向量库、结构化输出、文档解析等实验代码；
- 对比云服务组件、本地模型组件和传统算法在业务场景中的可行性；
- 为后续正式项目提供最小可运行样例和实现参考。

本项目不适合直接作为生产工程使用。部分示例依赖本地文件、外部 API key 或实验性路径，运行前需要按 README 调整环境变量和输入文件。

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── data/                         # 小型样例数据、FAISS 向量库样例、Flowgram 样例数据
├── docs/                         # PDF/压缩包/测试文档等输入资料
├── tests/                        # 结构整理与导入验证测试
└── src/agisample/
    ├── common/                   # 通用能力：连接、配置、路径等
    ├── langchain/                # LangChain 相关 demo
    │   ├── agents/               # Agent、Function Calling、工具调用
    │   ├── code/                 # 代码生成、代码评审等 LLM 编程样例
    │   ├── document/             # 文档/文件读取与 prompt 处理
    │   ├── extraction/           # 结构化输出、信息抽取、LangExtract
    │   ├── multimodal/           # 多模态/图像相关样例
    │   ├── rag/                  # RAG、重排序、问答链路
    │   ├── sql/                  # Text-to-SQL / Vanna 样例
    │   └── vectorstores/         # Embedding、FAISS、Elasticsearch 向量存储
    ├── langgraph/                # LangGraph 样例
    │   ├── basic/                # StateGraph 基础图、Flowgram 编译等入门样例
    │   ├── customersupport/      # 客服机器人：航班、酒店、租车等工具编排
    │   ├── hierarchical_teams/   # 分层多 Agent 团队（web/document 团队）
    │   └── a2a/                  # 双 Agent A2A 协议双向互调 demo
    ├── agentscope/               # AgentScope 框架样例
    ├── document_ai/              # 文档智能、OCR、发票/汇款/简历附件处理
    │   ├── azure_document_intelligence/
    │   └── resume_downloader/
    ├── local_models/             # 本地模型和 Gradio demo，例如 MiniCPM RAG
    ├── integrations/             # Langflow 等外部组件集成
    ├── machine_learning/         # 传统机器学习、匹配、降维等 demo
    ├── tools/                    # 通用工具脚本（组合查找、Markdown 恢复、邮件信息处理）
    ├── generic/                  # 旧路径兼容层
    └── base/                     # 旧路径兼容层
```

> 说明：原 `framework/` 目录已按主题完成迁移并删除——图示例迁入 `langgraph/`（`basic/`、`customersupport/`、`hierarchical_teams/`），信息抽取迁入 `langchain/extraction/`，代码与文档处理迁入 `langchain/code/`、`langchain/document/`，传统 ML 迁入 `machine_learning/`，通用工具迁入 `tools/`。

## 功能示例索引

| 主题 | 推荐入口 | 说明 | 主要依赖 |
| --- | --- | --- | --- |
| LangChain Agent | `src/agisample/langchain/agents/sample_agent_process.py` | 工具调用、数学函数、Agent 执行器示例 | LangChain、OpenAI 兼容接口 |
| JSON 参数 Agent | `src/agisample/langchain/agents/sample_agent_process_by_json.py` | 使用 JSON 字符串作为工具参数的 Agent 示例 | LangChain、OpenAI 兼容接口 |
| RAG 问答 | `src/agisample/langchain/rag/sample_rag_process.py` | PDF 加载、切分、FAISS 检索和 LLM 回答 | LangChain、FAISS、OpenAI Embeddings |
| 向量库管理 | `src/agisample/langchain/vectorstores/sample_data_vector_manager.py` | FAISS 本地向量库保存、加载、检索 | LangChain Community、FAISS |
| Elasticsearch 向量存储 | `src/agisample/langchain/vectorstores/sample_data_es_manager.py` | ElasticsearchStore 连接样例 | Elasticsearch、LangChain |
| 结构化输出 | `src/agisample/langchain/extraction/sample_structured_output_process.py` | Pydantic 结构化输出和 ReAct 风格样例 | LangChain、Pydantic |
| LangGraph 基础图 | `src/agisample/langgraph/basic/sample_graph_basic_chatbot.py` | 基础 StateGraph chatbot 示例 | LangGraph、LangChain |
| LangGraph 客服机器人 | `src/agisample/langgraph/customersupport/sample_graph_customer_support_bot_final.py` | 航班、酒店、租车、旅行推荐工具编排 | LangGraph、SQLite、LangChain |
| Flowgram 编译 | `src/agisample/langgraph/basic/sample_graph_by_flowgram3.py` | 从 flow.json 编译执行图 | LangGraph、Jinja2 |
| A2A 双向互调 | `src/agisample/langgraph/a2a/client.py`（先运行同目录 `server_a.py`、`server_b.py`） | 两个 LangGraph ReAct Agent 通过 A2A 协议双向互调，hop 上限防死循环 | LangGraph、a2a-sdk、uvicorn |
| Azure 文档读取 | `src/agisample/document_ai/azure_document_intelligence/prebuilt_read.py` | Azure Document Intelligence Read 模型示例 | Azure Document Intelligence |
| Azure 发票分析 | `src/agisample/document_ai/azure_document_intelligence/prebuilt_invoice.py` | Azure 预构建发票模型示例 | Azure Document Intelligence |
| 简历附件下载 | `src/agisample/document_ai/resume_downloader/download_resume_file.py` | 分页查询、解析附件 ID、下载文件 | requests |
| AgentScope | `src/agisample/agentscope/sample_agentscope.py` | AgentScope Agent、工具确认、流式输出示例 | AgentScope |
| MiniCPM RAG | `src/agisample/local_models/mini_cpm_rag.py` | 本地模型/Gradio 风格 RAG 示例 | MiniCPM、Gradio |
| Langflow 集成 | `src/agisample/integrations/langflow_process.py` | Langflow 调用样例 | Langflow |
| 降维示例 | `src/agisample/ml/dimensionality_reduction/sample_reduce_dimension_process.py` | 传统机器学习降维实验 | scikit-learn 等 |

## 环境准备

建议使用 Python 3.11 或以上版本。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前 `requirements.txt` 只覆盖了部分核心依赖。某些实验脚本还需要按需安装可选依赖，例如：

- Azure Document Intelligence：`azure-ai-documentintelligence`、`azure-core`
- 文档解析：`pymupdf`、`pdfplumber`、`pymupdf4llm`、`markitdown`
- 本地模型/前端：`gradio`、对应模型依赖
- 传统 ML：`scikit-learn`、`torch`、`transformers`
- LangChain v1 生态：`langchain`、`langchain-openai`、`langchain-community`、`langchain-elasticsearch`、`langgraph`、按需使用的 `langchain-classic` / `langchain-experimental`
- Agent 互调：`a2a-sdk`（已沉淀到 `requirements.txt`）
- 其他工具：`agentscope`、`vanna`、`cohere`

建议在运行具体 demo 前，根据 import 报错补充依赖，或把稳定需要的依赖逐步沉淀到 `requirements.txt`。

## 环境变量

在项目根目录创建 `.env` 文件，按需配置以下变量。不要提交真实 token。

```env
# OpenAI 兼容接口
OPENAI_API_KEY="sk-xxx"
OPENAI_BASE_URL="https://example.com/v1"

# AgentScope 示例
API_KEY="sk-xxx"

# Elasticsearch
ELASTICSEARCH_URL="http://localhost:9200"
ELASTICSEARCH_NAME="elastic"
ELASTICSEARCH_PASSWORD="password"

# Azure Document Intelligence
DI_KEY="your-document-intelligence-key"
DI_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"

# Tavily / Web Search（LangGraph 多 Agent 示例按需使用）
TAVILY_API_KEY="tvly-xxx"
```

代码中通常通过以下方式加载：

```python
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
```

## 快速运行

运行前建议先设置 `PYTHONPATH=src`，确保源码包可被导入：

```powershell
$env:PYTHONPATH='src'
```

### 1. 验证包结构

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe tests/test_package_structure.py -q
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
```

### 2. LangChain Agent 示例

```bash
.\\.venv\\Scripts\\python.exe src/agisample/langchain/agents/sample_agent_process.py
```

### 3. RAG 示例

```bash
.\\.venv\\Scripts\\python.exe src/agisample/langchain/rag/sample_rag_process.py
```

说明：RAG 示例依赖 `docs/llama2.pdf` 和 `data/sample_db`。如需重新构建向量库，可在代码中调用 `SampleRagProcess().init_data()`。

### 4. LangGraph 基础 Chatbot

```bash
.\\.venv\\Scripts\\python.exe src/agisample/langgraph/basic/sample_graph_basic_chatbot.py
```

### 5. Azure Document Intelligence 示例

```bash
PYTHONPATH=src python src/agisample/document_ai/azure_document_intelligence/prebuilt_read.py
PYTHONPATH=src python src/agisample/document_ai/azure_document_intelligence/prebuilt_invoice.py
```

说明：这两个脚本需要配置 `DI_KEY` 和 `DI_ENDPOINT`，并根据本地实际文件路径调整输入文件。

### 6. 简历附件下载示例

```bash
PYTHONPATH=src python src/agisample/document_ai/resume_downloader/download_resume_file.py \
  --token "your-token" \
  --status "Submitted （Unread）" \
  --page-size 50 \
  --out downloads
```

### 7. Deep Agents TypeScript 示例

`examples/langchain/deepagent/sample_deep_agent.ts` 是 JavaScript/TypeScript 生态示例，不属于 Python package，也不使用 `requirements.txt` 安装依赖。运行前请在单独的 Node.js 项目中安装 npm 依赖。

## 数据与生成产物

- `data/`：保存样例 JSON、Flowgram 输入、FAISS 索引等小型数据。
- `docs/`：保存 PDF、压缩包、测试文档等输入资料。
- 各示例运行产生的 `data_level0.bin`、`chroma.sqlite3` 等属于向量库或 Chroma 生成产物，建议统一放到 `data/` 或加入 `.gitignore`，避免源码目录混入运行时数据。
- `downloads/` 已在 `.gitignore` 中，用于保存下载脚本输出。

## 开发约定

新增 demo 时建议遵循以下规则：

1. 按主题选择目录：
   - LangChain 示例放到 `agisample/langchain/`；
   - LangGraph 示例放到 `agisample/langgraph/`；
   - 文档智能和文件解析放到 `agisample/document_ai/`；
   - 本地模型放到 `agisample/local_models/`；
   - 外部系统集成放到 `agisample/integrations/`；
   - 通用连接、路径、配置放到 `agisample/common/`。
2. 新文件优先使用 `snake_case.py` 命名。
3. 示例入口尽量提供 `if __name__ == "__main__":`，便于直接运行。
4. 不要在源码目录下写入大体积缓存、索引或下载文件。
5. 新增稳定依赖后同步更新 `requirements.txt`。
6. 涉及外部服务的 demo 不要提交真实凭证和公司内部数据。

## 旧路径兼容说明

原 `framework/` 目录已完成迁移并删除，旧的 `agisample.framework.*` import 不再可用，请改用新路径，例如：

```python
from agisample.langchain.rag.sample_rag_process import SampleRagProcess
```

`generic/`、`base/` 目前仍保留为旧路径兼容层。后续当 README 和示例全部切换到新路径后，可以逐步删除 `generic/`、`base/` 中的兼容 shim。

## 后续整理建议

本轮已经完成 `framework/` 全量迁移并删除该目录（图示例迁入 `langgraph/`，抽取/代码/文档处理迁入 `langchain/`，传统 ML 迁入 `machine_learning/`，通用工具迁入 `tools/`）。后续可以继续分阶段处理：

1. 增加统一路径工具，例如 `agisample.common.paths.PROJECT_ROOT`，替代各文件中不同层级的 `Path(__file__).resolve().parents[...]`。
2. 梳理 `requirements.txt`，把核心依赖和可选依赖拆分为 `requirements.txt` / `requirements-dev.txt` / `requirements-optional.txt`。
3. 为关键 demo 增加 smoke test，避免迁移后导入路径和入口脚本失效。

## 参考笔记

### Function Calling

Function Calling 是增强 LLM 交互能力的重要技术。模型可以在对话中调用特定函数，以执行任务或获取外部数据。为了提升稳定性，可以结合传统对话系统中的意图识别、槽位提取、上下文管理和调用前确认：

1. 意图识别：判断用户要执行的任务。
2. 槽位提取：提取函数调用所需参数。
3. 上下文管理：维护多轮对话状态和历史信息。
4. 验证和确认：在调用前确认必要参数完整且意图正确。

### RAG

RAG（Retrieval-Augmented Generation）结合检索和生成：先从外部知识库中检索相关内容，再把检索结果作为上下文交给 LLM 生成答案。

典型流程：

1. 准备数据并清洗无效内容。
2. 切分文档并生成向量。
3. 使用 ES 和向量数据库分别检索。
4. 对向量检索结果重排序。
5. 融合 ES 和向量检索结果。
6. 将问题和上下文组装为 prompt，提交给大模型。

### Agent

Agent 示例主要关注：

- 工具调用是否稳定；
- 参数是否能被可靠抽取；
- 执行失败时如何捕获异常；
- 是否需要人工反馈来修正执行过程；
- 如何结合短期记忆和 RAG 提供上下文。

### 方法论

LLM 幻觉无法完全避免。demo 的价值不只是“让模型回答”，还包括通过可解释的检索、结构化输出、校验规则和可观察的中间步骤，帮助用户识别和降低幻觉风险。
