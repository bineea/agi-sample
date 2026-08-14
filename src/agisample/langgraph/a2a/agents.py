"""两个对话 agent 的 system prompt（英文）与构建函数。"""

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from agisample.langgraph.a2a.llm_factory import build_chat_model
from agisample.langgraph.a2a.peer_tool import MAX_HOP, make_ask_peer_tool

AGENT_A_URL = "http://localhost:10001"
AGENT_B_URL = "http://localhost:10002"

_SYSTEM_PROMPT_TEMPLATE = (
    "You are {self_name}, a friendly chat agent in a two-agent A2A demo. "
    "Answer simple questions directly by yourself. "
    "When the user explicitly asks you to consult {peer_name}, or a second "
    "opinion would clearly help, call the {tool_name} tool with the question. "
    "The user message carries the current hop count as '[A2A hop=N]'; when "
    "calling the tool you MUST pass hop=N+1. The hard limit is {max_hop} "
    "agent-to-agent calls per request; if the tool says the limit is reached, "
    "stop calling it and answer with what you already know."
)

SYSTEM_PROMPT_A = _SYSTEM_PROMPT_TEMPLATE.format(
    self_name="Agent A", peer_name="Agent B", tool_name="ask_agentb",
    max_hop=MAX_HOP,
)
SYSTEM_PROMPT_B = _SYSTEM_PROMPT_TEMPLATE.format(
    self_name="Agent B", peer_name="Agent A", tool_name="ask_agenta",
    max_hop=MAX_HOP,
)


def build_agent_a() -> CompiledStateGraph:
    """构建 AgentA：挂载 ask_agentb 工具的 ReAct agent。"""
    return create_agent(
        model=build_chat_model(),
        tools=[make_ask_peer_tool("AgentB", AGENT_B_URL)],
        system_prompt=SYSTEM_PROMPT_A,
    )


def build_agent_b() -> CompiledStateGraph:
    """构建 AgentB：挂载 ask_agenta 工具的 ReAct agent。"""
    return create_agent(
        model=build_chat_model(),
        tools=[make_ask_peer_tool("AgentA", AGENT_A_URL)],
        system_prompt=SYSTEM_PROMPT_B,
    )
