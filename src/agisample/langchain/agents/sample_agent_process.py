import traceback

from dotenv import load_dotenv, find_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def multiply(first_int: int, second_int: int) -> int:
    """两个整数相乘。"""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """两个整数相加。"""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """计算 base 的 exponent 次方。"""
    return base**exponent


tools = [multiply, add, exponentiate]


def build_agent():
    load_dotenv(find_dotenv())
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, model_kwargs={"seed": 20})
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="你是一个可以调用数学工具回答问题的助手。请在需要计算时调用工具，并给出最终答案。",
    )


def main() -> None:
    agent = build_agent()
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": "1024的16倍是多少"}]})
        print(result["messages"][-1].content)
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()

