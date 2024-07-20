import json
import traceback

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

_ = load_dotenv(find_dotenv())


@tool
def multiply(multiply_json_data: str) -> int:
    """multiply two integers. """
    multiply_data = json.loads(multiply_json_data)
    return multiply_data['first_int'] * multiply_data['second_int']


@tool
def add(add_json_data: str) -> int:
    """Add two integers."""
    add_data = json.loads(add_json_data)
    return add_data['first_int'] * add_data['second_int']


@tool
def exponentiate(exponentiate_json_data: str) -> int:
    """Exponentiate the base to the exponent power."""
    exponentiate_data = json.loads(exponentiate_json_data)
    return exponentiate_data['base'] ** exponentiate_data['exponent']


print(multiply('''{"first_int":5,"second_int":3}'''))

tools = [multiply, add, exponentiate]

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, model_kwargs={"seed": 20})

PROMPT = """You are an AI assistant that can use tools to answer questions. You have access to the following tools:

{tools}

When answering a question, you should follow this process:

1. Think about what the question is asking.
2. Choose the appropriate tool to use.
3. Parse the necessary integers from the question.
4. Analyze the parameters needed for the chosen tool.
5. Format the integers obtained from the problem analysis into JSON format parameters for the selected tool
6. Observe the result from the tool.
7. Repeat if necessary to get the final answer.
8. Provide the final answer.

Use the following format:

Question: {input}
Thought: What do I need to do to answer the question?
Action: Which tool in [{tool_names}] to use and with what parameters.
Action Input: Format the integers obtained from the problem analysis into JSON format parameters for the selected tool.
Observation: The result from the tool.
... (this Thought/Action/Action Input/Observation sequence can repeat N times if needed)
Thought: What is the final answer?
Final Answer: The final answer to the original input question.Done!

Example:
Question: What is 5 multiplied by 3?
Thought: I need to multiply 5 by 3.
Action: multiply
Action Input: {{\"first_int\":5,\"second_int\":3\\}}
Observation: 15
Thought: I now know the final answer.
Final Answer: 15.

Now, answer the following question:

Question: {input}
Thought:{agent_scratchpad}"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(PROMPT)

# 定义一个 agent: 需要大模型、工具集、和 Prompt 模板
agent = create_react_agent(llm, tools, PROMPT_TEMPLATE)
# 定义一个执行器：需要 agent 对象 和 工具集
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

try:
    # 执行，并直接输出最终结果
    result = agent_executor.invoke({"input": "2的10次方是多少"})
    print(result)
except Exception as e:
    traceback.print_exc()
