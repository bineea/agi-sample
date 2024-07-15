import traceback

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

_ = load_dotenv(find_dotenv())


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers"""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """两个整数相加"""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """Exponentiate the base to the exponent power."""
    return base**exponent


tools = [multiply, add, exponentiate]

# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, model_kwargs={"seed":20})
llm = ChatOpenAI(model_name='gpt-4o', temperature=0, model_kwargs={"seed":20})


PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input should be parsed into the action's corresponding parameter
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# PROMPT = """You are an AI assistant that can use tools to answer questions. You have access to the following tools:
#
# {tools}
#
# When answering a question, you should follow this process:
#
# 1. Think about what the question is asking.
# 2. Choose the appropriate tool to use.
# 3. Parse the necessary integers from the question.
# 4. Use the tool with the parsed integers.
# 5. Observe the result from the tool.
# 6. Repeat if necessary to get the final answer.
# 7. Provide the final answer.
#
# Use the following format:
#
# Question: {input}
# Thought: What do I need to do to answer the question?
# Action: Which tool in [{tool_names}] to use and with what parameters.
# Action Input: The input parameters for the tool.
# Observation: The result from the tool.
# ... (this Thought/Action/Action Input/Observation sequence can repeat N times if needed)
# Thought: What is the final answer?
# Final Answer: The final answer to the original input question.Done!
#
# Example:
# Question: What is 5 multiplied by 3?
# Thought: I need to multiply 5 by 3.
# Action: multiply
# Action Input: 5, 3
# Observation: 15
# Thought: I now know the final answer.
# Final Answer: 15.
#
# Now, answer the following question:
#
# Question: {input}
# Thought:{agent_scratchpad}"""


PROMPT_TEMPLATE = ChatPromptTemplate.from_template(PROMPT)

# 定义一个 agent: 需要大模型、工具集、和 Prompt 模板
agent = create_react_agent(llm, tools, PROMPT_TEMPLATE)
# 定义一个执行器：需要 agent 对象 和 工具集
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

try:
    # 执行
    result = agent_executor.invoke({"input": "1024的16倍是多少"})
    print(result)
except Exception as e:
    traceback.print_exc()