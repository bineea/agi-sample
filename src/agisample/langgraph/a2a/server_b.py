"""AgentB 的 A2A 服务入口：监听 10002 端口。

运行方式（需先启动 server_a.py 才能演示完整互调链路）：
    $env:PYTHONPATH='src'
    .venv\\Scripts\\python.exe src/agisample/langgraph/a2a/server_b.py
"""

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agisample.langgraph.a2a.agent_executor import LangChainAgentExecutor
from agisample.langgraph.a2a.agents import AGENT_B_URL, build_agent_b

PORT = 10002


def build_agent_card() -> AgentCard:
    """构建 AgentB 的对外能力卡片（暴露在 /.well-known/agent-card.json）。"""
    skill = AgentSkill(
        id="chat",
        name="对话问答",
        description="日常聊天与简单问答，可通过 A2A 向 AgentA 求助",
        tags=["chat", "demo"],
        examples=["你好", "帮我问问 AgentA：今天天气怎么样？"],
    )
    return AgentCard(
        name="AgentB",
        description="A2A 双向互调 demo 中的 Agent B（LangGraph ReAct agent）",
        url=AGENT_B_URL + "/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


def main() -> None:
    application = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=DefaultRequestHandler(
            agent_executor=LangChainAgentExecutor(build_agent_b()),
            task_store=InMemoryTaskStore(),
        ),
    )
    print(f"AgentB 服务已启动: {AGENT_B_URL}（按 Ctrl+C 停止）")
    uvicorn.run(application.build(), host="localhost", port=PORT)


if __name__ == "__main__":
    main()
