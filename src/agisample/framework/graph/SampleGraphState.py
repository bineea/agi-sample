from langchain_core.tools import tool


@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return ["The answer to your question lies within."]


tools = [search]


from langgraph.prebuilt import ToolExecutor


tool_executor = ToolExecutor(tools)

import httpx
from contextlib import contextmanager
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig


class AgentContext(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    httpx_session: httpx.Client


@contextmanager
def make_agent_context(config: RunnableConfig):
    # here you could read the config values passed invoke/stream to customize the context object

    # as an example, we create an httpx session, which could then be used in your graph's nodes
    session = httpx.Client()
    try:
        yield AgentContext(httpx_session=session)
    finally:
        session.close()


import operator
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.channels.context import Context


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # 已经通过@contextmanager实现AbstractContextManager封装，忽略代码提示即可
    context: Annotated[AgentContext, Context(make_agent_context)]