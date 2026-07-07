# LangChain 最新版迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步实现。步骤使用 checkbox（`- [ ]`）跟踪。

## Context

用户要求检查项目中所有 LangChain 相关代码，并将代码与依赖升级到最新版本。当前项目是 Python demo 仓库，依赖入口只有 `requirements.txt`，目前锁定在 LangChain 0.3 / LangGraph 0.5 系列；代码中同时存在旧版 LangChain Agent/Chain/Retriever API、LangChain Community 中已拆分的 OpenAI 集成、LangGraph 旧 API，以及一个未跟踪的 `src/agisample/langchain/deepagent/sample_deep_agent.py`，其内容实际是 TypeScript/JavaScript 但扩展名为 `.py`。本计划目标是在保持现有 `requirements.txt` 管理方式的前提下，尽量迁移到 LangChain v1 / LangGraph v1 推荐写法，而不是只用 `langchain-classic` 兼容旧 API。

**Goal:** 将项目内 LangChain / LangGraph 相关 Python 代码迁移到最新 LangChain v1 生态，并同步升级 LangChain 相关依赖。

**Architecture:** 采用分阶段迁移：先升级依赖与低风险 import，再迁移 RAG/vectorstore/structured-output/LCEL，随后迁移 Agent 与 LangGraph，最后处理 deepagent 非 Python 文件与 README/测试。对无法低风险重写的 legacy API 仅在必要时临时引入 `langchain-classic`，并将使用范围限制在少数代表性 demo。

**Tech Stack:** Python 3.11+ / `requirements.txt` / LangChain v1 / LangGraph v1 / `langchain-openai` / `langchain-community` / `langchain-elasticsearch` / Pydantic v2 / pytest / compileall。

## Global Constraints

- 不引入 `pyproject.toml` / `uv.lock`，除非用户后续明确要求迁移到 uv project。
- 继续维护 `requirements.txt` 作为依赖源。
- 本机 Python 版本敏感操作不得使用裸 `python`；创建 venv 用 `py -3.13 -m venv .venv`，兼容旧项目时用 `py -3.11 -m venv .venv`。
- 安装依赖必须使用项目虚拟环境：`.\.venv\Scripts\python.exe -m pip ...`。
- 不提交或写入真实 API key；外部服务 smoke test 仅在用户已有 `.env` 且确认可调用时执行。
- 示例代码改动优先保持 demo 语义，不做无关大规模重构。
- Windows PowerShell 验证命令优先使用 `$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe ...`。

---

## 推荐方案与取舍

### 推荐方案：真正迁移到 v1 风格，少量 classic 兜底

- OpenAI 模型与 Embeddings 统一从 `langchain_openai` 导入。
- Chain 迁移到 LCEL：`prompt | model | parser`。
- Agent 迁移到 `langchain.agents.create_agent`；结构化输出优先使用 `with_structured_output()`。
- Retriever 迁移到 `.invoke(query)`。
- LangGraph 使用最新文档中的 `StateGraph`、`ToolNode`、`InMemorySaver`。
- `langchain-classic` 只允许用于一时无法替换的 legacy retriever/compressor，如 `LLMListwiseRerank`；若可在本轮重写 rerank，则不引入。

不推荐的替代方案：只升级依赖版本并把旧 import 全部改到 `langchain_classic`。这样改动少，但代码仍停留在旧范式，不符合“升级 LangChain 代码为最新版本”的目标。

---

## 文件结构与职责

### 依赖与文档

- Modify: `requirements.txt`
  - 升级 LangChain 生态版本。
  - 增加拆分后的集成包：`langchain-elasticsearch`、必要时 `langchain-experimental`、必要时 `langchain-classic`。
- Modify: `README.md`
  - 将安装/验证命令改为符合本机约束的 venv 命令。
  - 更新 LangChain v1、LangGraph v1、可选依赖说明。

### LangChain 核心示例

- Modify: `src/agisample/langchain/agents/sample_agent_process.py`
- Modify: `src/agisample/langchain/agents/sample_agent_process_by_json.py`
  - 旧 `create_react_agent + AgentExecutor` 迁移到 `create_agent` / structured output。
- Modify: `src/agisample/langchain/rag/sample_rag_process.py`
  - `ChatOpenAI` import、PDF load/split、LCEL RAG 保持并验证。
- Modify: `src/agisample/langchain/vectorstores/sample_data_vector_manager.py`
  - `OpenAIEmbeddings` import、retriever `.invoke()`、类型注解。
- Modify: `src/agisample/langchain/vectorstores/sample_data_es_manager.py`
  - `ElasticsearchStore` 迁移到 `langchain_elasticsearch`。
- Modify: `src/agisample/langchain/extraction/sample_structured_output_process.py`
  - `langchain_core.pydantic_v1` 迁移到 Pydantic v2。
- Modify: `src/agisample/langchain/multimodal/sample_image_process.py`
  - `LLMChain` / `langchain.llms.OpenAI` / `.run()` 迁移到 LCEL。
- Modify: `src/agisample/langchain/rag/sample_llm_rerank_process.py`
  - 优先重写为 v1 风格 rerank；如成本过高，临时迁到 `langchain_classic` 并显式记录。

### LangGraph / framework 中的 LangChain 代码

- Modify: `src/agisample/framework/graph/SampleGraphBasicChatbot.py`
- Modify: `src/agisample/framework/graph/SampleGraphProcess.py`
- Modify: `src/agisample/framework/graph/SampleGraphState.py`
- Modify: `src/agisample/framework/graph/SampleGraphChatbot.py`
- Modify: `src/agisample/framework/graph/SampleGraphAddHumanFeedback.py`
- Modify: `src/agisample/framework/graph/customersupport/*.py`
- Modify: `src/agisample/framework/graph/hierarchical_agent_teams/*.py`
  - 统一 OpenAI import、Pydantic import。
  - `ToolExecutor` 迁移到 `ToolNode`。
  - `MemorySaver` 优先迁移到 `InMemorySaver`。
  - `bind_functions` / `JsonOutputFunctionsParser` 改为 `with_structured_output()` 或 `bind_tools()`。

### framework/match 中的 LangChain 调用

- Modify representative files under `src/agisample/framework/match/`:
  - `LLMFileProcess.py`
  - `LLMGenerateCodeProcess.py`
  - `LLMGenerateCodeProcessBak.py`
  - `LLMReviewCodeProcess.py`
  - `LLMMultiExtractProcess*.py`
  - `LLMPromptProcess.py`
- 重点模式：
  - `_TextTemplateParam` / `_ImageTemplateParam` 是内部 API，改为公开 dict content block。
  - `get_relevant_documents()` 改为 `.invoke()`。
  - `load_and_split()` 改为 `load()` + splitter。

### deepagent 未跟踪目录

- Modify or move: `src/agisample/langchain/deepagent/sample_deep_agent.py`
- Recommended handling:
  - 若保留为 JS/TS 示例：改扩展名为 `.ts`，移到 `examples/langchain/deepagent/sample_deep_agent.ts`，不作为 Python package 的一部分。
  - 若用户后续明确要 Python deep agent 示例：重写为 Python `create_agent` / LangGraph 示例。
- 本轮推荐默认采用“移出 Python package 并改为 `.ts` 示例”，因为原文件本身是 TS/JS 内容。

---

## Task 1: 建立升级基线并更新 LangChain 依赖

**Files:**
- Modify: `requirements.txt`
- Read/verify: `README.md`

**Interfaces:**
- Consumes: 当前 `requirements.txt` 中 LangChain 0.3 / LangGraph 0.5 版本。
- Produces: 可安装且无依赖冲突的最新 LangChain v1 依赖集合。

**Steps:**

- [ ] 查看当前工作区状态，确认用户已有未跟踪 deepagent 目录存在：
  ```powershell
  git status --short
  ```
  Expected: 看到 `?? src/agisample/langchain/deepagent/` 或其他用户已有改动；不要覆盖无关改动。

- [ ] 查询执行时 PyPI 最新版本，并以执行时结果为准更新版本；探索阶段参考版本为：
  ```text
  langchain-core==1.4.8
  langchain==1.3.11
  langchain-openai==1.3.3
  langchain-text-splitters==1.1.2
  langchain-community==0.4.2
  langgraph==1.2.8
  langsmith==0.9.8
  langchain-classic==1.0.8  # 仅必要时
  ```

- [ ] 更新 `requirements.txt`：
  - 保留 `python-dotenv`。
  - 升级 LangChain 生态包到执行时最新兼容版本。
  - 增加 `langchain-elasticsearch`。
  - 如果 `document_tool.py` 继续保留 `PythonREPL`，增加 `langchain-experimental`。
  - 如果 `sample_llm_rerank_process.py` 暂不重写，增加 `langchain-classic`。

- [ ] 使用项目虚拟环境安装并检查依赖：
  ```powershell
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe -m pip check
  ```
  Expected: 安装成功，`pip check` 无冲突。若 `.venv` 不存在，先征得用户同意后用 `py -3.13 -m venv .venv` 创建。

---

## Task 2: 迁移低风险 LangChain import 与基础 API

**Files:**
- Modify: `src/agisample/langchain/rag/sample_rag_process.py`
- Modify: `src/agisample/langchain/vectorstores/sample_data_vector_manager.py`
- Modify: `src/agisample/langchain/vectorstores/sample_data_es_manager.py`
- Modify: `src/agisample/langchain/extraction/sample_structured_output_process.py`
- Modify representative framework graph files using old imports.

**Interfaces:**
- Produces: 最新导入路径与 retriever API，供后续 RAG、Agent、Graph 任务复用。

**Steps:**

- [ ] 将所有 `from langchain_community.chat_models import ChatOpenAI` 改为：
  ```python
  from langchain_openai import ChatOpenAI
  ```
  代表文件：`sample_agent_process.py`、`sample_agent_process_by_json.py`、`sample_rag_process.py`、`SampleGraphBasicChatbot.py`、`SampleGraphProcess.py`。

- [ ] 将所有 `from langchain_community.embeddings import OpenAIEmbeddings` 改为：
  ```python
  from langchain_openai import OpenAIEmbeddings
  ```
  代表文件：`sample_data_vector_manager.py`、`sample_data_es_manager.py`。

- [ ] 将 `langchain_core.pydantic_v1` 改为 Pydantic v2：
  ```python
  from pydantic import BaseModel, Field
  ```
  如果某文件只导入 `BaseModel`，则使用：
  ```python
  from pydantic import BaseModel
  ```

- [ ] 将 retriever 查询从旧方法改为 runnable 调用：
  ```python
  docs = self.retriever().invoke(query)
  ```
  代表文件：`sample_data_vector_manager.py`、`framework/match/LLMFileProcess.py`。

- [ ] 将 `PyMuPDFLoader.load_and_split()` 改为显式 load + split：
  ```python
  documents = pdf_loader.load()
  split_docs = text_splitter.split_documents(documents)
  ```
  代表文件：`sample_rag_process.py`、`framework/match/LLMFileProcess.py`。

- [ ] 运行导入与编译验证：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe tests/test_package_structure.py -q
  ```
  Expected: 无 SyntaxError；导入测试不因 import 路径失败。

---

## Task 3: 迁移 RAG 与 VectorStore 集成

**Files:**
- Modify: `src/agisample/langchain/rag/sample_rag_process.py`
- Modify: `src/agisample/langchain/vectorstores/sample_data_vector_manager.py`
- Modify: `src/agisample/langchain/vectorstores/sample_data_es_manager.py`
- Modify: `src/agisample/common/elasticsearch_connection.py` only if constructor compatibility requires it.

**Interfaces:**
- Produces: `SampleDataVectorManager.retriever()` 仍返回 retriever runnable；`search(query)` 返回首个文档内容。

**Steps:**

- [ ] 修正 `SampleDataVectorManager.save` 的类型注解：
  ```python
  def save(self, documents: list[Document]) -> None:
      faiss_client = FAISS.from_documents(documents, self.__embedding)
      faiss_client.save_local(
          os.path.join(SampleDataVectorManager.BASE_DIR, "data", SampleDataVectorManager.DB_LOCAL_FILE_NAME),
          SampleDataVectorManager.INDEX_NAME,
      )
  ```

- [ ] 保留 FAISS community 导入，除非执行时官方包拆分要求变化：
  ```python
  from langchain_community.vectorstores import FAISS
  ```

- [ ] 将 ElasticsearchStore 导入改为：
  ```python
  from langchain_elasticsearch import ElasticsearchStore
  ```
  然后按最新版构造签名调整 `sample_data_es_manager.py` 中 `ElasticsearchStore(...)` 参数。

- [ ] 验证 import smoke test：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "from agisample.langchain.vectorstores.sample_data_vector_manager import SampleDataVectorManager; print(SampleDataVectorManager.__name__)"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "from agisample.langchain.rag.sample_rag_process import SampleRagProcess; print(SampleRagProcess.__name__)"
  ```
  Expected: 打印类名，不触发外部 API 调用。

---

## Task 4: 将 legacy Chain 与结构化输出迁移到 v1 写法

**Files:**
- Modify: `src/agisample/langchain/multimodal/sample_image_process.py`
- Modify: `src/agisample/langchain/extraction/sample_structured_output_process.py`
- Modify representative `src/agisample/framework/match/*.py` files.

**Interfaces:**
- Produces: 使用 LCEL 的 chain；Pydantic v2 schema 可被 `with_structured_output()` 使用。

**Steps:**

- [ ] 在 `sample_image_process.py` 中移除：
  ```python
  from langchain.chains import LLMChain
  from langchain.llms import OpenAI
  from langchain.prompts import PromptTemplate
  ```

- [ ] 替换为：
  ```python
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_openai import ChatOpenAI
  ```

- [ ] 将 chain 构造改为：
  ```python
  prompt = ChatPromptTemplate.from_template(template)
  llm = ChatOpenAI(model="gpt-4o", temperature=0)
  chain = prompt | llm | StrOutputParser()
  result = chain.invoke({"text": text})
  ```

- [ ] 将 framework/match 中的内部 prompt 类型替换为公开 dict content block，例如：
  ```python
  [
      {"type": "text", "text": "这是文档当前页面的文本内容：{content}"},
      {"type": "text", "text": "这是显示文档当前页面内容的图像："},
      {"type": "image_url", "image_url": {"url": "data:image/JPEG;base64,{image}"}},
  ]
  ```
  替换目标包括 `_TextTemplateParam(...)` 和 `_ImageTemplateParam(...)`。

- [ ] 如果 Pydantic v2 对 `.dict()` / `.json()` 报错，改为：
  ```python
  model.model_dump()
  model.model_dump_json()
  ```

- [ ] 运行编译验证：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
  ```

---

## Task 5: 迁移 Agent 示例到 `create_agent`

**Files:**
- Modify: `src/agisample/langchain/agents/sample_agent_process.py`
- Modify: `src/agisample/langchain/agents/sample_agent_process_by_json.py`

**Interfaces:**
- Produces: 可直接运行的 LangChain v1 Agent 示例；工具函数继续为 `multiply`、`add`、`exponentiate`。

**Steps:**

- [ ] 将旧导入：
  ```python
  from langchain.agents import create_react_agent, AgentExecutor
  ```
  替换为：
  ```python
  from langchain.agents import create_agent
  ```

- [ ] 将 `ChatOpenAI(model_name=...)` 改为：
  ```python
  llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, model_kwargs={"seed": 20})
  ```

- [ ] 用 `create_agent` 构造 agent：
  ```python
  agent = create_agent(
      model=llm,
      tools=tools,
      system_prompt="你是一个可以调用数学工具回答问题的助手。请在需要计算时调用工具，并给出最终答案。",
  )
  ```

- [ ] 将调用输入改为 v1 messages 格式：
  ```python
  result = agent.invoke({
      "messages": [
          {"role": "user", "content": "1024的16倍是多少"}
      ]
  })
  print(result["messages"][-1].content)
  ```

- [ ] 对 JSON 参数示例，优先改成结构化工具参数，而不是要求模型生成 JSON 字符串；例如工具签名保持：
  ```python
  @tool
  def multiply(first_int: int, second_int: int) -> int:
      """两个整数相乘。"""
      return first_int * second_int
  ```

- [ ] 验证：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.agents.sample_agent_process"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.agents.sample_agent_process_by_json"
  ```
  Expected: import 不执行真实 LLM 调用；如当前文件仍顶层执行，则先加 `main()` 和 `if __name__ == '__main__':`。

---

## Task 6: 迁移 LangGraph 相关代码

**Files:**
- Modify: `src/agisample/framework/graph/SampleGraphState.py`
- Modify: `src/agisample/framework/graph/SampleGraphBasicChatbot.py`
- Modify: `src/agisample/framework/graph/SampleGraphProcess.py`
- Modify: `src/agisample/framework/graph/SampleGraphChatbot.py`
- Modify: `src/agisample/framework/graph/customersupport/*.py`
- Modify: `src/agisample/framework/graph/hierarchical_agent_teams/*.py`

**Interfaces:**
- Produces: 可在 LangGraph v1 下 import/compile 的 graph 示例。

**Steps:**

- [ ] 将 `ToolExecutor` 替换为 `ToolNode`，并检查 state 是否传递 messages：
  ```python
  from langgraph.prebuilt import ToolNode

  tool_node = ToolNode(tools)
  ```

- [ ] 将新示例中短期内存保存器统一为：
  ```python
  from langgraph.checkpoint.memory import InMemorySaver

  memory = InMemorySaver()
  ```

- [ ] 将 `langchain_community.chat_models.ChatOpenAI` 改为 `langchain_openai.ChatOpenAI`。

- [ ] 将 `bind_functions` + `JsonOutputFunctionsParser` 路由改为 Pydantic schema + `with_structured_output()`：
  ```python
  from typing import Literal
  from pydantic import BaseModel, Field

  class RouteResponse(BaseModel):
      next: Literal["FINISH", "Search", "WebScraper"] = Field(description="下一个要执行的节点")
  ```
  然后用：
  ```python
  router = llm.with_structured_output(RouteResponse)
  route = router.invoke(messages)
  ```
  实际 Literal 值按文件中的成员名调整。

- [ ] 将交互循环或真实调用放入 main guard，避免 import 测试阻塞：
  ```python
  def main() -> None:
      ...

  if __name__ == "__main__":
      main()
  ```

- [ ] 验证 LangGraph import：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.framework.graph.SampleGraphState"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.framework.graph.SampleGraphBasicChatbot"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.framework.graph.customersupport.SampleGraphCustomerSupportBotFinal"
  ```

---

## Task 7: 处理 legacy rerank 与 classic 依赖

**Files:**
- Modify: `src/agisample/langchain/rag/sample_llm_rerank_process.py`
- Modify: `requirements.txt` if classic is required.

**Interfaces:**
- Produces: 最新依赖下可 import 的 rerank 示例。

**Steps:**

- [ ] 优先尝试用 v1 结构化输出重写 rerank：定义 Pydantic 输出 schema，输入 query 与 documents，输出排序后的索引/理由。

- [ ] 如果该重写超出本轮范围，则临时使用：
  ```python
  from langchain_classic.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank
  ```
  并在 `requirements.txt` 中添加 `langchain-classic` 的执行时最新兼容版本。

- [ ] 修正当前断言与测试数据不一致的问题，避免仍检查旧的 `Steve` 示例。

- [ ] 验证：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.rag.sample_llm_rerank_process"
  ```

---

## Task 8: 处理 deepagent TS/JS `.py` 文件

**Files:**
- Move/Rename: `src/agisample/langchain/deepagent/sample_deep_agent.py`
- Potentially create: `examples/langchain/deepagent/sample_deep_agent.ts`
- Potentially modify: `README.md`

**Interfaces:**
- Produces: Python package 中不再包含非法 Python 语法文件。

**Steps:**

- [ ] 默认按 TS 示例处理：将 `sample_deep_agent.py` 移到：
  ```text
  examples/langchain/deepagent/sample_deep_agent.ts
  ```

- [ ] 保留原内容但更新 JS LangChain 最新导入时，按执行时 JS 文档确认；不要把 npm 依赖加入 Python `requirements.txt`。

- [ ] 如果用户要求 Python deepagent 示例，则不要移动为 TS，而是重写为 Python `create_agent` 示例。

- [ ] 验证 Python 编译不再扫描到非法 `.py`：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
  ```

---

## Task 9: 更新 README 与测试/验证脚本

**Files:**
- Modify: `README.md`
- Modify: `tests/test_package_structure.py` if new/renamed modules require adjustment.

**Interfaces:**
- Produces: README 中的安装、验证、LangChain 示例说明与新代码一致。

**Steps:**

- [ ] 将 README 中裸 `python` / `pip` 安装示例更新为 Windows 友好的虚拟环境命令：
  ```powershell
  py -3.13 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  ```

- [ ] 将验证命令更新为：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe tests/test_package_structure.py -q
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
  ```

- [ ] 更新依赖说明：LangChain v1、LangGraph v1、`langchain-openai`、`langchain-elasticsearch`、可选 `langchain-classic` / `langchain-experimental`。

- [ ] 如果 deepagent 移到 `examples/`，在 README 说明这是 JS/TS 示例，不属于 Python package。

---

## Task 10: 最终验证

**Files:**
- No code changes expected unless verification exposes issues.

**Steps:**

- [ ] 运行依赖检查：
  ```powershell
  .\.venv\Scripts\python.exe -m pip check
  ```
  Expected: 无依赖冲突。

- [ ] 运行编译检查：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
  ```
  Expected: 无 SyntaxError。

- [ ] 运行结构测试：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe tests/test_package_structure.py -q
  ```
  Expected: PASS。

- [ ] 运行代表性 import smoke test：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.agents.sample_agent_process"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.rag.sample_rag_process"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.vectorstores.sample_data_vector_manager"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.langchain.extraction.sample_structured_output_process"
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import agisample.framework.graph.SampleGraphState"
  ```

- [ ] 如 `.env` 中已有可用 API key，并且用户确认可以调用外部服务，再运行少量真实 demo：
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe src/agisample/langchain/agents/sample_agent_process.py
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe src/agisample/langchain/rag/sample_rag_process.py
  ```
  Expected: demo 正常返回；若因缺少 API key 或外部资源失败，如实记录为环境问题而非代码验证通过。

---

## 风险与缓解

- **Agent v1 迁移不是机械替换。** `create_react_agent + AgentExecutor` 的 prompt、输入、输出都要改。缓解：先迁移简单数学工具示例，再处理 hierarchical team。
- **`langchain-classic` 可能掩盖旧代码。** 缓解：只允许少数 legacy rerank 文件临时使用，其他 chain/agent 必须迁到 v1/LCEL。
- **LangGraph `ToolExecutor` -> `ToolNode` 需要检查 state schema。** 缓解：逐文件 import smoke test，再跑最小 graph invoke。
- **Pydantic v2 行为变化。** 缓解：检查 `.dict()` / `.json()` 并改为 `.model_dump()` / `.model_dump_json()`。
- **deepagent `.py` 实际是 TS/JS。** 缓解：移出 Python package 或重写为 Python；不能保留非法 `.py`。
- **部分 demo 依赖可选包或外部 API。** 缓解：最终验证区分 import/compile 级别与真实外部服务级别。

## Self-Review

- Spec coverage: 覆盖依赖升级、LangChain core 示例、RAG/vectorstore、structured output、legacy chain、agent、LangGraph、framework/match、deepagent、README 与验证。
- Placeholder scan: 无 TBD/TODO/“后续补充”占位；每个任务都有具体文件与命令。
- Type consistency: `SampleDataVectorManager.save(documents: list[Document]) -> None`、retriever `.invoke(query)`、`create_agent` messages 输入在任务间保持一致。
