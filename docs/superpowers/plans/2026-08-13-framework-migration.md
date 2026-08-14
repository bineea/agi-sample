# framework 目录整体迁移与删除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步实现。步骤使用 checkbox（`- [ ]`）跟踪。

**Goal:** 把 `src/agisample/framework/` 下所有仍有价值的文件按性质迁移到 `langchain/`、`langgraph/`、`machine_learning/`、`tools/`,统一 snake_case 命名,重写内部 import,最后彻底删除 `framework/` 目录。

**Architecture:** 纯文件迁移 + 重命名 + import 重写,不改变任何运行时行为。按目标目录分任务,每个任务迁移一组文件并即时验证。用 `git mv` 保留历史。真正的交叉引用只有 11 个文件,集中处理。

**Tech Stack:** Python 3.11+ / git mv / pytest(unittest) / compileall / importlib.find_spec。

**Spec:** `docs/superpowers/specs/2026-08-13-framework-migration-design.md`

## Global Constraints

- 不引入 `pyproject.toml` / `uv.lock`;继续只维护 `requirements.txt`(本计划不改依赖)。
- 不使用裸 `python`;验证一律用 `.\.venv\Scripts\python.exe`。
- Windows 验证命令统一 PowerShell 形式:`$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe ...`。
- 迁移文件改 snake_case 文件名,**类名保持 CamelCase 不变**。
- 不改任何代码逻辑,只移动文件、改文件名、重写 import 路径。
- 每个任务结束跑 compileall + find_spec 验证。
- 本计划**不删** `LLMGenerateCodeProcessBak.py`(用户确认保留);只删空目录 `hierarchical_customer_support/`。

## 内部交叉引用全量(仅 11 个文件,重写依据)

模块路径映射(包路径 + CamelCase 模块名一起改):

| 旧 import 路径 | 新 import 路径 |
|---|---|
| `agisample.framework.graph.customersupport.InitDatabase` | `agisample.langgraph.customersupport.init_database` |
| `agisample.framework.graph.customersupport.BookCar` | `agisample.langgraph.customersupport.book_car` |
| `agisample.framework.graph.customersupport.BookHotel` | `agisample.langgraph.customersupport.book_hotel` |
| `agisample.framework.graph.customersupport.FetchFlightInfo` | `agisample.langgraph.customersupport.fetch_flight_info` |
| `agisample.framework.graph.customersupport.LookupPolicy` | `agisample.langgraph.customersupport.lookup_policy` |
| `agisample.framework.graph.customersupport.SupportTrip` | `agisample.langgraph.customersupport.support_trip` |
| `agisample.framework.graph.customersupport.LLMModel` | `agisample.langgraph.customersupport.llm_model` |
| `agisample.framework.graph.customersupport.HandleTool` | `agisample.langgraph.customersupport.handle_tool` |
| `agisample.framework.graph.hierarchical_agent_teams.<mod>` | `agisample.langgraph.hierarchical_teams.<mod>`(`<mod>` 不变:document_tool/handle_agent/document_team/web_team/web_tool) |
| `agisample.framework.match.RecoveryToMarkdown` | `agisample.tools.recovery_to_markdown` |

含这些引用的文件:customersupport 的 `BookHotel/FetchFlightInfo/LLMModel/SupportTrip/SampleGraphCustomerSupportBot/SampleGraphCustomerSupportBotFinal`;hierarchical_agent_teams 的 `document_team/handle_agent/main_team/web_team`;match 的 `LLMFileProcess`。

---

## Task 1: 创建目标目录骨架

**Files:**
- Create: `src/agisample/langgraph/basic/__init__.py`
- Create: `src/agisample/langgraph/customersupport/__init__.py`
- Create: `src/agisample/langgraph/hierarchical_teams/__init__.py`
- Create: `src/agisample/langchain/code/__init__.py`
- Create: `src/agisample/langchain/document/__init__.py`
- Create: `src/agisample/tools/__init__.py`

**Interfaces:**
- Produces: 6 个空包目录,供后续任务迁入文件。`langchain/extraction/`、`machine_learning/` 已存在,无需新建。

**Steps:**

- [ ] 创建 6 个新目录,每个放一个空 `__init__.py`:
  ```powershell
  New-Item -ItemType Directory -Force src/agisample/langgraph/basic, src/agisample/langgraph/customersupport, src/agisample/langgraph/hierarchical_teams, src/agisample/langchain/code, src/agisample/langchain/document, src/agisample/tools
  New-Item -ItemType File -Force src/agisample/langgraph/basic/__init__.py, src/agisample/langgraph/customersupport/__init__.py, src/agisample/langgraph/hierarchical_teams/__init__.py, src/agisample/langchain/code/__init__.py, src/agisample/langchain/document/__init__.py, src/agisample/tools/__init__.py
  ```

- [ ] 验证目录可被 Python 识别为包:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import importlib.util as u; [print(p, u.find_spec(p) is not None) for p in ['agisample.langgraph.basic','agisample.langgraph.customersupport','agisample.langgraph.hierarchical_teams','agisample.langchain.code','agisample.langchain.document','agisample.tools']]"
  ```
  Expected: 全部 `True`。

- [ ] Commit:
  ```bash
  git add src/agisample/langgraph src/agisample/langchain/code src/agisample/langchain/document src/agisample/tools
  git commit -m "chore: 创建 framework 迁移目标目录骨架"
  ```

---

## Task 2: 迁移 langgraph/basic/(graph 顶层 8 个基础示例)

**Files:**
- Move: `src/agisample/framework/graph/SampleGraphState.py` → `src/agisample/langgraph/basic/sample_graph_state.py`
- Move: `src/agisample/framework/graph/SampleGraphBasicChatbot.py` → `src/agisample/langgraph/basic/sample_graph_basic_chatbot.py`
- Move: `src/agisample/framework/graph/SampleGraphChatbot.py` → `src/agisample/langgraph/basic/sample_graph_chatbot.py`
- Move: `src/agisample/framework/graph/SampleGraphProcess.py` → `src/agisample/langgraph/basic/sample_graph_process.py`
- Move: `src/agisample/framework/graph/SampleGraphAddHumanFeedback.py` → `src/agisample/langgraph/basic/sample_graph_add_human_feedback.py`
- Move: `src/agisample/framework/graph/SampleGraphByFlowgram1.py` → `src/agisample/langgraph/basic/sample_graph_by_flowgram1.py`
- Move: `src/agisample/framework/graph/SampleGraphByFlowgram2.py` → `src/agisample/langgraph/basic/sample_graph_by_flowgram2.py`
- Move: `src/agisample/framework/graph/SampleGraphByFlowgram3.py` → `src/agisample/langgraph/basic/sample_graph_by_flowgram3.py`

**Interfaces:**
- Produces: `agisample.langgraph.basic.<snake_case_name>` 可导入。这 8 个文件无 framework 内部交叉引用,无需改 import。

**Steps:**

- [ ] 用 `git mv` 逐个移动并重命名:
  ```bash
  git mv src/agisample/framework/graph/SampleGraphState.py src/agisample/langgraph/basic/sample_graph_state.py
  git mv src/agisample/framework/graph/SampleGraphBasicChatbot.py src/agisample/langgraph/basic/sample_graph_basic_chatbot.py
  git mv src/agisample/framework/graph/SampleGraphChatbot.py src/agisample/langgraph/basic/sample_graph_chatbot.py
  git mv src/agisample/framework/graph/SampleGraphProcess.py src/agisample/langgraph/basic/sample_graph_process.py
  git mv src/agisample/framework/graph/SampleGraphAddHumanFeedback.py src/agisample/langgraph/basic/sample_graph_add_human_feedback.py
  git mv src/agisample/framework/graph/SampleGraphByFlowgram1.py src/agisample/langgraph/basic/sample_graph_by_flowgram1.py
  git mv src/agisample/framework/graph/SampleGraphByFlowgram2.py src/agisample/langgraph/basic/sample_graph_by_flowgram2.py
  git mv src/agisample/framework/graph/SampleGraphByFlowgram3.py src/agisample/langgraph/basic/sample_graph_by_flowgram3.py
  ```

- [ ] 编译验证:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample/langgraph/basic
  ```
  Expected: 无 SyntaxError。

- [ ] find_spec 验证 8 个新模块可被找到:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import importlib.util as u; mods=['sample_graph_state','sample_graph_basic_chatbot','sample_graph_chatbot','sample_graph_process','sample_graph_add_human_feedback','sample_graph_by_flowgram1','sample_graph_by_flowgram2','sample_graph_by_flowgram3']; [print(m, u.find_spec('agisample.langgraph.basic.'+m) is not None) for m in mods]"
  ```
  Expected: 全部 `True`。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework/graph src/agisample/langgraph/basic
  git commit -m "refactor: graph 基础示例迁入 langgraph/basic"
  ```

---

## Task 3: 迁移 langgraph/customersupport/(10 个文件 + 内部 import 重写)

**Files:**
- Move: `src/agisample/framework/graph/customersupport/{BookCar,BookHotel,FetchFlightInfo,HandleTool,InitDatabase,LLMModel,LookupPolicy,SupportTrip,SampleGraphCustomerSupportBot,SampleGraphCustomerSupportBotFinal}.py` → `src/agisample/langgraph/customersupport/` 对应 snake_case 名
- Modify: 移动后的 `book_hotel.py, fetch_flight_info.py, llm_model.py, support_trip.py, sample_graph_customer_support_bot.py, sample_graph_customer_support_bot_final.py`(含内部 import)

**Interfaces:**
- Produces: `agisample.langgraph.customersupport.<snake_case>` 可导入;内部互相引用全部指向新包。

文件名映射:`BookCar→book_car, BookHotel→book_hotel, FetchFlightInfo→fetch_flight_info, HandleTool→handle_tool, InitDatabase→init_database, LLMModel→llm_model, LookupPolicy→lookup_policy, SupportTrip→support_trip, SampleGraphCustomerSupportBot→sample_graph_customer_support_bot, SampleGraphCustomerSupportBotFinal→sample_graph_customer_support_bot_final`。

**Steps:**

- [ ] `git mv` 移动 10 个文件到 `src/agisample/langgraph/customersupport/` 并改 snake_case 名:
  ```bash
  cd src/agisample
  git mv framework/graph/customersupport/BookCar.py langgraph/customersupport/book_car.py
  git mv framework/graph/customersupport/BookHotel.py langgraph/customersupport/book_hotel.py
  git mv framework/graph/customersupport/FetchFlightInfo.py langgraph/customersupport/fetch_flight_info.py
  git mv framework/graph/customersupport/HandleTool.py langgraph/customersupport/handle_tool.py
  git mv framework/graph/customersupport/InitDatabase.py langgraph/customersupport/init_database.py
  git mv framework/graph/customersupport/LLMModel.py langgraph/customersupport/llm_model.py
  git mv framework/graph/customersupport/LookupPolicy.py langgraph/customersupport/lookup_policy.py
  git mv framework/graph/customersupport/SupportTrip.py langgraph/customersupport/support_trip.py
  git mv framework/graph/customersupport/SampleGraphCustomerSupportBot.py langgraph/customersupport/sample_graph_customer_support_bot.py
  git mv framework/graph/customersupport/SampleGraphCustomerSupportBotFinal.py langgraph/customersupport/sample_graph_customer_support_bot_final.py
  cd ../../..
  ```

- [ ] 重写移动后文件里的内部 import(按"内部交叉引用全量"表)。逐条精确替换:

  `llm_model.py` 中:
  ```python
  from agisample.langgraph.customersupport.book_car import search_car_rentals, update_car_rental, book_car_rental, cancel_car_rental
  from agisample.langgraph.customersupport.book_hotel import search_hotels, book_hotel, update_hotel, cancel_hotel
  from agisample.langgraph.customersupport.fetch_flight_info import fetch_user_flight_information, search_flights, update_ticket_to_new_flight, cancel_ticket
  from agisample.langgraph.customersupport.lookup_policy import lookup_policy
  from agisample.langgraph.customersupport.support_trip import search_trip_recommendations, update_excursion, cancel_excursion, book_excursion
  ```

  `book_hotel.py` / `fetch_flight_info.py` / `support_trip.py` 中:
  ```python
  from agisample.langgraph.customersupport.init_database import db
  ```

  `sample_graph_customer_support_bot.py` 与 `sample_graph_customer_support_bot_final.py` 中:
  ```python
  from agisample.langgraph.customersupport.fetch_flight_info import fetch_user_flight_information
  from agisample.langgraph.customersupport.llm_model import part_1_assistant_runnable, part_1_safe_tools, part_1_sensitive_tools
  from agisample.langgraph.customersupport.handle_tool import create_tool_node_with_fallback, _print_event
  from agisample.langgraph.customersupport.init_database import backup_file, db
  ```
  (注:各文件原有 import 的具体名称以文件实际内容为准,只替换 `agisample.framework.graph.customersupport.X` → `agisample.langgraph.customersupport.x` 这一段,import 的符号名不变。)

- [ ] 确认 customersupport 下不再有旧引用:
  ```bash
  grep -rn "agisample.framework" src/agisample/langgraph/customersupport || echo "clean"
  ```
  Expected: 输出 `clean`。

- [ ] 编译 + find_spec 验证:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample/langgraph/customersupport
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import importlib.util as u; mods=['book_car','book_hotel','fetch_flight_info','handle_tool','init_database','llm_model','lookup_policy','support_trip','sample_graph_customer_support_bot','sample_graph_customer_support_bot_final']; [print(m, u.find_spec('agisample.langgraph.customersupport.'+m) is not None) for m in mods]"
  ```
  Expected: 编译无错;全部 `True`。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework/graph/customersupport src/agisample/langgraph/customersupport
  git commit -m "refactor: customersupport 迁入 langgraph/customersupport 并重写内部 import"
  ```

---

## Task 4: 迁移 langgraph/hierarchical_teams/(6 个文件 + 内部 import 重写)

**Files:**
- Move: `src/agisample/framework/graph/hierarchical_agent_teams/{document_team,document_tool,handle_agent,main_team,web_team,web_tool}.py` → `src/agisample/langgraph/hierarchical_teams/`(文件名不变)
- Modify: 移动后的 `document_team.py, handle_agent.py, main_team.py, web_team.py`(含内部 import)

**Interfaces:**
- Produces: `agisample.langgraph.hierarchical_teams.<mod>` 可导入,模块名不变。

**Steps:**

- [ ] `git mv` 移动 6 个文件(文件名已是 snake_case,不变):
  ```bash
  cd src/agisample
  git mv framework/graph/hierarchical_agent_teams/document_team.py langgraph/hierarchical_teams/document_team.py
  git mv framework/graph/hierarchical_agent_teams/document_tool.py langgraph/hierarchical_teams/document_tool.py
  git mv framework/graph/hierarchical_agent_teams/handle_agent.py langgraph/hierarchical_teams/handle_agent.py
  git mv framework/graph/hierarchical_agent_teams/main_team.py langgraph/hierarchical_teams/main_team.py
  git mv framework/graph/hierarchical_agent_teams/web_team.py langgraph/hierarchical_teams/web_team.py
  git mv framework/graph/hierarchical_agent_teams/web_tool.py langgraph/hierarchical_teams/web_tool.py
  cd ../../..
  ```

- [ ] 重写内部 import,统一把 `agisample.framework.graph.hierarchical_agent_teams.` 替换为 `agisample.langgraph.hierarchical_teams.`(模块名不变)。涉及:

  `handle_agent.py`:
  ```python
  from agisample.langgraph.hierarchical_teams.web_tool import create_tavily_tool, scrape_webpages
  ```
  `document_team.py`:
  ```python
  from agisample.langgraph.hierarchical_teams.document_tool import WORKING_DIRECTORY, write_document, edit_document, read_document, create_outline, python_repl
  from agisample.langgraph.hierarchical_teams.handle_agent import create_agent, agent_node, create_team_supervisor
  ```
  `web_team.py`:
  ```python
  from agisample.langgraph.hierarchical_teams.handle_agent import create_team_supervisor, agent_node, create_agent
  from agisample.langgraph.hierarchical_teams.web_tool import scrape_webpages, create_tavily_tool
  ```
  `main_team.py`:
  ```python
  from agisample.langgraph.hierarchical_teams.document_team import authoring_chain
  from agisample.langgraph.hierarchical_teams.handle_agent import create_team_supervisor
  from agisample.langgraph.hierarchical_teams.web_team import research_chain
  ```

- [ ] 确认无旧引用:
  ```bash
  grep -rn "agisample.framework" src/agisample/langgraph/hierarchical_teams || echo "clean"
  ```
  Expected: `clean`。

- [ ] 编译 + find_spec 验证:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample/langgraph/hierarchical_teams
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import importlib.util as u; mods=['document_team','document_tool','handle_agent','main_team','web_team','web_tool']; [print(m, u.find_spec('agisample.langgraph.hierarchical_teams.'+m) is not None) for m in mods]"
  ```
  Expected: 编译无错;全部 `True`。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework/graph/hierarchical_agent_teams src/agisample/langgraph/hierarchical_teams
  git commit -m "refactor: hierarchical_agent_teams 迁入 langgraph/hierarchical_teams"
  ```

---

## Task 5: 迁移 langchain/extraction/(match 的 11 个抽取示例)

**Files:**
- Move: `src/agisample/framework/match/LLMMultiExtractProcess.py` → `src/agisample/langchain/extraction/llm_multi_extract_process.py`
- Move: `src/agisample/framework/match/LLMMultiExtractProcessEmail1.py` → `.../llm_multi_extract_process_email1.py`(下同规则)
- Move: `LLMMultiExtractProcessEmail2.py` → `llm_multi_extract_process_email2.py`
- Move: `LLMMultiExtractProcessExcel1.py` → `llm_multi_extract_process_excel1.py`
- Move: `LLMMultiExtractProcessExcel2.py` → `llm_multi_extract_process_excel2.py`
- Move: `LLMMultiExtractProcessExcel3.py` → `llm_multi_extract_process_excel3.py`
- Move: `LLMMultiExtractProcessPdf1.py` → `llm_multi_extract_process_pdf1.py`
- Move: `LLMMultiExtractProcessPdf2.py` → `llm_multi_extract_process_pdf2.py`
- Move: `LLMGenerateExtractDocumentByClaude.py` → `llm_generate_extract_document_by_claude.py`
- Move: `LLMGenerateExtractDocumentByGPT4o.py` → `llm_generate_extract_document_by_gpt4o.py`
- Move: `LangExtractProcess.py` → `lang_extract_process.py`

**Interfaces:**
- Produces: `agisample.langchain.extraction.<snake_case>` 可导入。这些文件无 framework 内部交叉引用(经 grep 确认),无需改 import。目标目录已有的 `sample_lang_extract.py`、`sample_structured_output_process.py` 不动。

**Steps:**

- [ ] `git mv` 移动 11 个文件到 `src/agisample/langchain/extraction/` 并改 snake_case 名(逐条执行):
  ```bash
  cd src/agisample
  git mv framework/match/LLMMultiExtractProcess.py langchain/extraction/llm_multi_extract_process.py
  git mv framework/match/LLMMultiExtractProcessEmail1.py langchain/extraction/llm_multi_extract_process_email1.py
  git mv framework/match/LLMMultiExtractProcessEmail2.py langchain/extraction/llm_multi_extract_process_email2.py
  git mv framework/match/LLMMultiExtractProcessExcel1.py langchain/extraction/llm_multi_extract_process_excel1.py
  git mv framework/match/LLMMultiExtractProcessExcel2.py langchain/extraction/llm_multi_extract_process_excel2.py
  git mv framework/match/LLMMultiExtractProcessExcel3.py langchain/extraction/llm_multi_extract_process_excel3.py
  git mv framework/match/LLMMultiExtractProcessPdf1.py langchain/extraction/llm_multi_extract_process_pdf1.py
  git mv framework/match/LLMMultiExtractProcessPdf2.py langchain/extraction/llm_multi_extract_process_pdf2.py
  git mv framework/match/LLMGenerateExtractDocumentByClaude.py langchain/extraction/llm_generate_extract_document_by_claude.py
  git mv framework/match/LLMGenerateExtractDocumentByGPT4o.py langchain/extraction/llm_generate_extract_document_by_gpt4o.py
  git mv framework/match/LangExtractProcess.py langchain/extraction/lang_extract_process.py
  cd ../../..
  ```

- [ ] 编译 + find_spec 验证:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample/langchain/extraction
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import importlib.util as u; mods=['llm_multi_extract_process','llm_multi_extract_process_email1','llm_multi_extract_process_email2','llm_multi_extract_process_excel1','llm_multi_extract_process_excel2','llm_multi_extract_process_excel3','llm_multi_extract_process_pdf1','llm_multi_extract_process_pdf2','llm_generate_extract_document_by_claude','llm_generate_extract_document_by_gpt4o','lang_extract_process']; [print(m, u.find_spec('agisample.langchain.extraction.'+m) is not None) for m in mods]"
  ```
  Expected: 编译无错;全部 `True`。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework/match src/agisample/langchain/extraction
  git commit -m "refactor: match 抽取示例迁入 langchain/extraction"
  ```

---

## Task 6: 迁移 langchain/code/ 与 langchain/document/

**Files:**
- Move: `src/agisample/framework/match/LLMGenerateCodeProcess.py` → `src/agisample/langchain/code/llm_generate_code_process.py`
- Move: `src/agisample/framework/match/LLMGenerateCodeProcessBak.py` → `src/agisample/langchain/code/llm_generate_code_process_bak.py`
- Move: `src/agisample/framework/match/LLMReviewCodeProcess.py` → `src/agisample/langchain/code/llm_review_code_process.py`
- Move: `src/agisample/framework/match/LLMFileProcess.py` → `src/agisample/langchain/document/llm_file_process.py`
- Move: `src/agisample/framework/match/LLMPromptProcess.py` → `src/agisample/langchain/document/llm_prompt_process.py`
- Modify: 移动后的 `llm_file_process.py`(含对 RecoveryToMarkdown 的 import)

**Interfaces:**
- Produces: `agisample.langchain.code.{llm_generate_code_process,llm_generate_code_process_bak,llm_review_code_process}`、`agisample.langchain.document.{llm_file_process,llm_prompt_process}`。
- Consumes: `llm_file_process.py` 依赖 Task 7 才迁入的 `agisample.tools.recovery_to_markdown`;本任务先改成新路径,验证在 Task 7 完成后统一做。

**Steps:**

- [ ] `git mv` 移动 5 个文件:
  ```bash
  cd src/agisample
  git mv framework/match/LLMGenerateCodeProcess.py langchain/code/llm_generate_code_process.py
  git mv framework/match/LLMGenerateCodeProcessBak.py langchain/code/llm_generate_code_process_bak.py
  git mv framework/match/LLMReviewCodeProcess.py langchain/code/llm_review_code_process.py
  git mv framework/match/LLMFileProcess.py langchain/document/llm_file_process.py
  git mv framework/match/LLMPromptProcess.py langchain/document/llm_prompt_process.py
  cd ../../..
  ```

- [ ] 改 `llm_file_process.py` 第 40 行附近的 import:
  ```python
  from agisample.tools.recovery_to_markdown import convert_info_markdown
  ```
  (该路径在 Task 7 创建 `tools/recovery_to_markdown.py` 后才可解析,本任务编译时该 import 不影响 `compileall` 的语法检查。)

- [ ] 编译验证(仅语法,不执行 import):
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample/langchain/code src/agisample/langchain/document
  ```
  Expected: 无 SyntaxError。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework/match src/agisample/langchain/code src/agisample/langchain/document
  git commit -m "refactor: 代码/文档处理示例迁入 langchain/code 与 langchain/document"
  ```

---

## Task 7: 迁移 machine_learning/ 与 tools/

**Files:**
- Move: `src/agisample/framework/match/EncodeOnlyProcess.py` → `src/agisample/machine_learning/encode_only_process.py`
- Move: `src/agisample/framework/match/RandomForestProcess.py` → `src/agisample/machine_learning/random_forest_process.py`
- Move: `src/agisample/framework/match/GradientBoostingDecisionTreeProcess.py` → `src/agisample/machine_learning/gradient_boosting_decision_tree_process.py`
- Move: `src/agisample/framework/match/FindCombinations.py` → `src/agisample/tools/find_combinations.py`
- Move: `src/agisample/framework/match/RecoveryToMarkdown.py` → `src/agisample/tools/recovery_to_markdown.py`
- Move: `src/agisample/framework/EmailInfoProcess.py` → `src/agisample/tools/email_info_process.py`

**Interfaces:**
- Produces: `agisample.machine_learning.{encode_only_process,random_forest_process,gradient_boosting_decision_tree_process}`、`agisample.tools.{find_combinations,recovery_to_markdown,email_info_process}`。这些文件无 framework 内部交叉引用。
- 完成后 `agisample.tools.recovery_to_markdown` 就位,Task 6 的 `llm_file_process.py` import 可解析。

**Steps:**

- [ ] `git mv` 移动 6 个文件:
  ```bash
  cd src/agisample
  git mv framework/match/EncodeOnlyProcess.py machine_learning/encode_only_process.py
  git mv framework/match/RandomForestProcess.py machine_learning/random_forest_process.py
  git mv framework/match/GradientBoostingDecisionTreeProcess.py machine_learning/gradient_boosting_decision_tree_process.py
  git mv framework/match/FindCombinations.py tools/find_combinations.py
  git mv framework/match/RecoveryToMarkdown.py tools/recovery_to_markdown.py
  git mv framework/EmailInfoProcess.py tools/email_info_process.py
  cd ../../..
  ```

- [ ] 编译 + find_spec 验证(含 Task 6 的 `llm_file_process` 一并复查):
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample/machine_learning src/agisample/tools
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -c "import importlib.util as u; paths=['agisample.machine_learning.encode_only_process','agisample.machine_learning.random_forest_process','agisample.machine_learning.gradient_boosting_decision_tree_process','agisample.tools.find_combinations','agisample.tools.recovery_to_markdown','agisample.tools.email_info_process','agisample.langchain.document.llm_file_process']; [print(p, u.find_spec(p) is not None) for p in paths]"
  ```
  Expected: 编译无错;全部 `True`。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework src/agisample/machine_learning src/agisample/tools
  git commit -m "refactor: 传统 ML 与工具示例迁入 machine_learning 与 tools"
  ```

---

## Task 8: 删除 framework 目录并更新测试

**Files:**
- Delete: `src/agisample/framework/`(整个目录,含 13 个 shim、`__init__.py`、空的 `hierarchical_customer_support/`)
- Modify: `tests/test_package_structure.py`

**Interfaces:**
- Consumes: 前 7 个任务已把所有要保留的文件迁出。
- Produces: `framework` 包不再存在;测试只断言新路径。

**Steps:**

- [ ] 确认 framework 下已无任何待保留的 `.py`(只剩 shim 和空目录):
  ```bash
  find src/agisample/framework -name "*.py" | sort
  ```
  Expected: 只剩 13 个 `Sample*.py` shim、`__init__.py` 系列、以及 `graph/hierarchical_customer_support/__init__.py`。逐一确认它们都是 shim(头部含 `from agisample.langchain... import *`)或空文件后才继续。

- [ ] 删除整个 framework 目录:
  ```bash
  git rm -r src/agisample/framework
  ```

- [ ] 修改 `tests/test_package_structure.py`:
  - 从 `LEGACY_IMPORT_PATHS` 列表中**删除**全部 9 个 `agisample.framework.*` 条目(即 `SampleAgentProcess`、`SampleAgentProcessByJson`、`SampleRagProcess`、`SampleDataVectorManager`、`SampleDataEsManager`、`SampleSQLProcess`、`SampleImageProcess`、`SampleStructuredOutputProcess`、`SampleAgentScope`)。保留 `agisample.base.*` 和 `agisample.generic.*` 条目不动。
  - 向 `NEW_IMPORT_PATHS` 列表**追加**以下新路径:
    ```python
    "agisample.langgraph.basic.sample_graph_state",
    "agisample.langgraph.basic.sample_graph_basic_chatbot",
    "agisample.langgraph.basic.sample_graph_chatbot",
    "agisample.langgraph.basic.sample_graph_process",
    "agisample.langgraph.basic.sample_graph_add_human_feedback",
    "agisample.langgraph.customersupport.llm_model",
    "agisample.langgraph.customersupport.sample_graph_customer_support_bot_final",
    "agisample.langgraph.hierarchical_teams.main_team",
    "agisample.langgraph.hierarchical_teams.web_tool",
    "agisample.langchain.extraction.llm_multi_extract_process",
    "agisample.langchain.extraction.lang_extract_process",
    "agisample.langchain.code.llm_generate_code_process",
    "agisample.langchain.code.llm_review_code_process",
    "agisample.langchain.document.llm_file_process",
    "agisample.langchain.document.llm_prompt_process",
    "agisample.machine_learning.encode_only_process",
    "agisample.machine_learning.random_forest_process",
    "agisample.tools.find_combinations",
    "agisample.tools.recovery_to_markdown",
    "agisample.tools.email_info_process",
    ```

- [ ] 运行结构测试:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe tests/test_package_structure.py -q
  ```
  Expected: PASS(新路径可找到;framework 旧路径断言已移除,不会因目录删除而失败)。

- [ ] 确认全仓库不再有 framework 引用:
  ```bash
  grep -rn "agisample.framework\|agisample\.framework" src tests --include="*.py" || echo "clean"
  ```
  Expected: `clean`(README 在下一任务处理)。

- [ ] Commit:
  ```bash
  git add -A src/agisample/framework tests/test_package_structure.py
  git commit -m "refactor: 删除 framework 目录,测试切换到新路径"
  ```

---

## Task 9: 更新 README 并最终验证

**Files:**
- Modify: `README.md`

**Steps:**

- [ ] 找到 README 中所有 framework 旧引用并改为新路径。已知 `README.md:213`:
  ```python
  from agisample.framework.SampleRagProcess import SampleRagProcess
  ```
  改为:
  ```python
  from agisample.langchain.rag.sample_rag_process import SampleRagProcess
  ```
  并全文件搜索 `agisample.framework`、`framework/` 字样,逐一替换为对应新路径(若该示例已迁到 langchain/langgraph,用新的 snake_case 路径)。
  ```bash
  grep -n "framework" README.md
  ```

- [ ] 全量编译检查:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m compileall -q src/agisample
  ```
  Expected: 无 SyntaxError。

- [ ] 结构测试最终确认:
  ```powershell
  $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe tests/test_package_structure.py -q
  ```
  Expected: PASS。

- [ ] 代表性 import smoke test(哑 key,验证纯 LangChain/LangGraph 模块可导入,不发真实请求):
  ```powershell
  $env:PYTHONPATH='src'; $env:OPENAI_API_KEY='dummy'; $env:TAVILY_API_KEY='dummy'; .\.venv\Scripts\python.exe -c "import agisample.langgraph.basic.sample_graph_state, agisample.langgraph.hierarchical_teams.web_tool, agisample.langgraph.hierarchical_teams.handle_agent; print('langgraph OK')"
  ```
  Expected: 打印 `langgraph OK`。

- [ ] 确认 `framework/` 已被 git 完全移除:
  ```bash
  git status --short
  ls src/agisample/framework 2>&1 || echo "framework removed"
  ```
  Expected: 无 `src/agisample/framework` 残留;输出 `framework removed`。

- [ ] Commit:
  ```bash
  git add README.md
  git commit -m "docs: README 更新 framework 迁移后的新路径"
  ```

---

## Self-Review

- **Spec coverage:** 覆盖 spec 的全部映射(basic/customersupport/hierarchical_teams/extraction/code/document/machine_learning/tools)、删除项(仅空目录 + framework 树)、test/README 更新、验证。`LLMGenerateCodeProcessBak` 按用户确认保留并迁入 code/。
- **Placeholder scan:** 无 TBD/TODO;每个任务的文件、git mv 命令、import 重写内容、验证命令均具体给出。
- **Type consistency:** 内部 import 重写统一遵循"内部交叉引用全量"表;customersupport 的 `part_1_assistant_runnable/part_1_safe_tools/part_1_sensitive_tools`、`create_tool_node_with_fallback/_print_event`、`backup_file/db`、hierarchical_teams 的 `create_tavily_tool/scrape_webpages/authoring_chain/research_chain` 等符号名迁移前后保持不变(只改模块路径)。
