"""演示客户端：向 AgentA 发起问题，观察 A↔B 双向互调链路。

运行前请先启动 server_a.py 与 server_b.py，然后：
    $env:PYTHONPATH='src'
    .venv\\Scripts\\python.exe src/agisample/langgraph/a2a/client.py
"""

import asyncio

from agisample.langgraph.a2a.agents import AGENT_A_URL
from agisample.langgraph.a2a.peer_tool import ask_peer_text

# 三个递进问题，分别对应：不互调 → A→B 单向 → A↔B 双向（观察 hop 上限）
DEMO_QUESTIONS = [
    "你好，请介绍一下你自己。",
    "请帮我问问 AgentB：1 加 1 等于几？并把它的回答告诉我。",
    "请和 AgentB 互相确认一下：法国的首都是哪里？",
]


async def main() -> None:
    print("A2A 双向互调演示客户端（请确认 server_a.py 与 server_b.py 已启动）\n")
    for question in DEMO_QUESTIONS:
        print(f"客户端 → AgentA: {question}")
        # client 首发 hop=0；ask_peer_text 内部把失败转成中文提示
        reply = await ask_peer_text(question, AGENT_A_URL, hop=0)
        if reply.startswith("无法连接"):
            print(f"{reply}请先运行 server_a.py。\n")
            break
        print(f"AgentA 回复: {reply}\n")
    print("演示结束。")


if __name__ == "__main__":
    asyncio.run(main())
