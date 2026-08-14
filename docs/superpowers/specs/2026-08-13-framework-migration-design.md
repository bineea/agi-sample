# framework 目录整体迁移与删除 设计

> 日期:2026-08-13
> 状态:待评审
> 关联:`docs/superpowers/plans/2026-07-07-langchain-v1-migration.md`(迁移后续)

## 背景与目标

LangChain v1 迁移后,`src/agisample/framework/` 是最后一批未归位的旧代码。该目录里实际是三类内容:

- 顶层 14 个 `Sample*.py`:13 个已是纯转发 shim(真实代码在 `agisample.langchain.*`),仅 `EmailInfoProcess.py` 为独立实现
- `framework/graph/`:LangGraph 示例
- `framework/match/`:LLM 文档/代码处理 + 少量传统 ML + 纯工具

**目标:彻底删除 `framework/` 目录**,把所有仍有价值的内容按性质迁移到对应的分类目录,不保留兼容 shim,并顺势删除死代码。

项目已有惯例(见 `tests/test_package_structure.py`):真实代码放分类目录、snake_case 命名。本设计沿用该惯例,但**不再保留 framework 旧路径**。

## 关键决策(已与用户确认)

| 决策点 | 结论 |
|---|---|
| 重组目标 | 彻底删除 framework,不留 shim |
| 迁移方式 | 分类迁移 + 顺势清理(方案 3) |
| 非 LLM 工具文件 | 保留,迁到新建 `tools/` |
| 迁移后命名 | 统一 snake_case |
| `LLMMultiExtractProcess` 7 个变体 | **保留迁移**(经核实是针对不同文档类型的不同示例,非重复代码) |

## 目标结构

```
src/agisample/
├── langgraph/
│   ├── basic/                 # graph/ 顶层 8 个基础示例
│   ├── customersupport/       # graph/customersupport/
│   └── hierarchical_teams/    # graph/hierarchical_agent_teams/
├── langchain/
│   ├── extraction/            # match/ 的 *Extract* 系列(并入现有目录)
│   ├── code/                  # 新建:代码生成/审查
│   └── document/              # 新建:文档加载/提示
├── machine_learning/          # match/ 的传统 ML(并入现有目录)
└── tools/                     # 新建:与 LLM 无关的工具示例
```

每个新目录含空 `__init__.py`。迁移文件改 snake_case,类名保持 CamelCase 不变。

## 文件映射

### langgraph/basic/(来自 framework/graph/ 顶层)
| 源 | 目标 |
|---|---|
| SampleGraphState.py | sample_graph_state.py |
| SampleGraphBasicChatbot.py | sample_graph_basic_chatbot.py |
| SampleGraphChatbot.py | sample_graph_chatbot.py |
| SampleGraphProcess.py | sample_graph_process.py |
| SampleGraphAddHumanFeedback.py | sample_graph_add_human_feedback.py |
| SampleGraphByFlowgram1.py | sample_graph_by_flowgram1.py |
| SampleGraphByFlowgram2.py | sample_graph_by_flowgram2.py |
| SampleGraphByFlowgram3.py | sample_graph_by_flowgram3.py |

### langgraph/customersupport/(来自 framework/graph/customersupport/)
BookCar→book_car, BookHotel→book_hotel, FetchFlightInfo→fetch_flight_info, HandleTool→handle_tool, InitDatabase→init_database, LLMModel→llm_model, LookupPolicy→lookup_policy, SupportTrip→support_trip, SampleGraphCustomerSupportBot→sample_graph_customer_support_bot, SampleGraphCustomerSupportBotFinal→sample_graph_customer_support_bot_final

### langgraph/hierarchical_teams/(来自 framework/graph/hierarchical_agent_teams/)
已是 snake_case,文件名不变:document_team, document_tool, handle_agent, main_team, web_team, web_tool

### langchain/extraction/(来自 framework/match/)
LLMMultiExtractProcess→llm_multi_extract_process,及 7 个变体(Email1/Email2/Excel1/Excel2/Excel3/Pdf1/Pdf2)同规则 snake_case;LLMGenerateExtractDocumentByClaude→llm_generate_extract_document_by_claude,LLMGenerateExtractDocumentByGPT4o→llm_generate_extract_document_by_gpt4o,LangExtractProcess→lang_extract_process

### langchain/code/(新建)
LLMGenerateCodeProcess→llm_generate_code_process, LLMGenerateCodeProcessBak→llm_generate_code_process_bak, LLMReviewCodeProcess→llm_review_code_process

### langchain/document/(新建)
LLMFileProcess→llm_file_process, LLMPromptProcess→llm_prompt_process

### machine_learning/(并入现有)
EncodeOnlyProcess→encode_only_process, RandomForestProcess→random_forest_process, GradientBoostingDecisionTreeProcess→gradient_boosting_decision_tree_process

### tools/(新建)
FindCombinations→find_combinations, RecoveryToMarkdown→recovery_to_markdown, EmailInfoProcess→email_info_process(来自 framework 顶层)

### 删除
- `framework/graph/hierarchical_customer_support/`(空目录,仅空 `__init__.py`)
- `framework/` 顶层 13 个 shim、`__init__.py`,以及迁移完成后的整个 `framework/` 目录

> 注:`LLMGenerateCodeProcessBak.py` 经用户确认**保留迁移**,不删除。

## 连带改动

1. **内部 import 重写**:framework 内约 76 处 `agisample.framework.*` 互相引用,全部改为新路径(如 `agisample.langgraph.hierarchical_teams.web_tool`)。
2. **`tests/test_package_structure.py`**:
   - 从 `LEGACY_IMPORT_PATHS` 删除全部 `agisample.framework.*` 条目(其余 `agisample.base.*` / `agisample.generic.*` 保留,不在本次范围)。
   - 向 `NEW_IMPORT_PATHS` 增加各新模块路径(注:测试用 `importlib.util.find_spec`,只验证可找到、不执行顶层代码,因此含顶层 `ChatOpenAI()` 的模块也可安全列入)。
3. **`README.md`**:更新 `from agisample.framework.SampleRagProcess import ...`(约 line 213)等 framework 引用为新路径。

## 不在本次范围

- 顶层实例化 `ChatOpenAI()` / `input()` 导致的 import 期真实调用(类别 C,用户已明确不做 main guard)。
- 依赖外部服务/重型可选包的模块(ES、torch、talon、vanna、cohere、ppocr、camelot 等),按环境性问题处理,不在迁移中修复。
- 两个含硬编码本机路径的抽取示例(`LLMGenerateExtractDocumentByClaude/GPT4o`):迁移保留,路径问题属配置项,不修。
- 其他旧目录(`base/`、`generic/` 等)。

## 验证

1. `compileall -q src/agisample` 无语法错误。
2. `tests/test_package_structure.py` 通过(新路径可被 find_spec 找到,framework 旧路径条目已移除)。
3. 代表性 import smoke test(哑 API key 下验证纯 LangChain/LangGraph 模块可导入,不发真实请求)。
4. `git status` 确认 `framework/` 已完全移除。

## 风险与缓解

- **内部引用遗漏**:76 处交叉 import,逐一 grep `agisample.framework` 确认清零。
- **命名冲突**:迁移后与现有 `langchain/extraction/`、`machine_learning/` 内文件重名时调整。
- **删除误伤**:仅删除空目录;`LLMGenerateCodeProcessBak` 经用户确认保留,其余一律迁移。
