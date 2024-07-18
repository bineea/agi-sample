# AGI

## 配置Openai
由于国内与openai的双向封锁，建议直接申请使用DevAGI的服务（https://devcto.com）

## 设置环境变量
1. 项目根目录创建".env"文件
2. 配置变量 </br>
``OPENAI_API_KEY="sk-xxx"`` </br>
``OPENAI_BASE_URL="https://api.fe8.cn/v1"``
3. 加载变量 </br>
``from dotenv import load_dotenv, find_dotenv`` </br>
``_ = load_dotenv(find_dotenv())``
4. 读取变量 </br>
``ELASTICSEARCH_BASE_URL = os.getenv('ELASTICSEARCH_BASE_URL')``

## 生成requestments.txt
1. 安装pipreqs
``pip install pipreqs``
2. 生成requestment
``pipreqs ./ --encoding=utf8 --force``

## Function Calling 
在大型语言模型（LLM，Large Language Models）如OpenAI的GPT-4中，Function Calling 是一种增强模型功能和交互能力的重要技术。这种技术允许语言模型在对话中调用特定的函数，以便执行特定的任务或获取外部数据。

Function Calling完全依赖LLM决定是否触发函数调用，所以通过槽位（slots）和意图（intent）分析可以增加Function Calling的稳定性。这种方法来源于传统的对话系统设计，用于确保在合适的上下文中触发正确的函数。
1. 意图识别
   * 意图分类器：使用机器学习模型或规则系统来识别用户的意图。这可以基于用户输入的自然语言文本来判断用户的需求。 
   * 预训练模型：像BERT等模型可以用来提高意图分类的准确性。
2. 槽位提取
   * 实体识别：使用命名实体识别（NER）技术从用户输入中提取参数或槽位。这些槽位是函数调用所需的参数。 
   * 模板匹配：基于预定义的模式匹配来提取槽位
3. 上下文管理
   * 对话状态追踪：维护对话状态，记录用户的意图和槽位。这有助于处理多轮对话和不完整的信息。
   * 历史对话记忆：记录用户先前的输入，确保在需要时可以访问之前的上下文。
4. 验证和确认
   * 槽位确认：在调用函数前，确认所有必要的槽位都已填充。如果某些槽位缺失，可以通过澄清问题来获取。
   * 意图确认：在调用函数前，确认识别的意图是否正确。可以通过向用户提问来确认。

## RAG
RAG（Retrieval-Augmented Generation）是一种结合检索（Retrieval）和生成（Generation）技术的方法，用于改进自然语言处理（NLP）任务中的文本生成。具体来说，RAG模型首先从一个大型文档集合中检索相关文档，然后使用这些文档作为参考，通过生成模型生成回答或内容。这样的方法能够提高生成文本的准确性和相关性，特别是在需要具体事实和详细信息的任务中。以下是RAG的工作原理：

1. **检索阶段（Retrieval Stage）**：
   - 使用一个检索模型（通常是基于BERT或其他预训练语言模型的双塔架构）从一个文档集合中找到与输入查询最相关的文档。
   - 这些文档被用作生成阶段的辅助信息。

2. **生成阶段（Generation Stage）**：
   - 使用一个生成模型（通常是基于Transformer架构的生成模型，如GPT）结合检索到的文档生成回答或内容。
   - 生成模型通过将输入查询和检索到的文档一起输入，生成更为准确和相关的文本。

这种方法的优势在于它能够利用外部知识库（例如维基百科）中的信息，从而生成的回答不仅仅依赖于模型的内部知识，而是可以参考外部的、最新的、详细的信息。RAG特别适用于问答系统、对话系统和需要动态更新信息的生成任务。

### RAG实现步骤
1. 准备数据通过LLM做优化，减小无效词，比如：语气词
2. 通过LLM优化精简明确问题描述
3. 通过ES和向量数据库分别基于问题描述查询数据 
4. 针对向量数据库返回结果做重排序 
5. 针对ES和向量数据库返回结果做融合排序 
6. 将问题描述和融合排序后的查询结果整合prompt，提交给大模型

## AGENT
???


### THINK
结合rag，可以先通过rag找到对应的文件或数据？？？因为调用操作是LLM执行的，如何能把rag结合？
agent_executor invoke 需要try catch捕捉异常
如果允许LLM提问，比如LLM执行错误时，由人工来引导，怎么改造这个prompt；但是也可以不用提问，因为保留了短时记忆，由人工继续提问即可

### METHODOLOGY
* 大模型的幻觉无法避免，通过100%精确的算法解释大模型执行过程，让非技术用户也能快速识别幻觉，做到“可信”
