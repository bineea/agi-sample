import operator
from typing import Annotated, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel


@tool
def search(query: str):
    """Call to surf the web."""
    return ["The answer to your question lies within."]


tools = [search]
tool_node = ToolNode(tools)


class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]
