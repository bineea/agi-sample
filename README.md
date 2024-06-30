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