"""A2A 互调工具的离线单元测试：不需要真实 LLM 与对端服务。"""

import asyncio
import unittest

from agisample.langgraph.a2a.agent_executor import _to_text
from agisample.langgraph.a2a.peer_tool import (
    MAX_HOP,
    _extract_reply_text,
    current_hop,
    make_ask_peer_tool,
)
from a2a.types import Message, Part, Role, TextPart

# 一个必定无人监听的地址，用于模拟"对方服务未启动"
DEAD_PEER_URL = "http://localhost:19999"


class PeerToolHopLimitTest(unittest.TestCase):
    """hop 超过上限时工具应直接短路，不发网络请求。"""

    def test_hop_over_limit_short_circuits(self):
        tool = make_ask_peer_tool("AgentB", DEAD_PEER_URL)
        reply = asyncio.run(tool.ainvoke({"question": "你好", "hop": MAX_HOP + 1}))
        self.assertIn("已达到最大互调次数上限", reply)

    def test_contextvar_overrides_lying_llm(self):
        # LLM 谎报 hop=1，但本进程 contextvar 已是 MAX_HOP，仍应被拦下
        tool = make_ask_peer_tool("AgentB", DEAD_PEER_URL)
        token = current_hop.set(MAX_HOP)
        try:
            reply = asyncio.run(tool.ainvoke({"question": "你好", "hop": 1}))
        finally:
            current_hop.reset(token)
        self.assertIn("已达到最大互调次数上限", reply)


class PeerToolDegradationTest(unittest.TestCase):
    """对方服务不可达时应返回降级提示而非抛异常。"""

    def test_dead_peer_returns_graceful_message(self):
        tool = make_ask_peer_tool("AgentB", DEAD_PEER_URL)
        reply = asyncio.run(tool.ainvoke({"question": "你好", "hop": 1}))
        self.assertIn("无法连接对方服务", reply)


class ReplyExtractTest(unittest.TestCase):
    """A2A 响应文本提取与 AIMessage content 归一化。"""

    def test_extract_text_from_message(self):
        message = Message(
            role=Role.agent,
            message_id="m1",
            parts=[Part(root=TextPart(text="你好")), Part(root=TextPart(text="世界"))],
        )
        self.assertEqual(_extract_reply_text(message), "你好世界")

    def test_to_text_handles_str_and_blocks(self):
        self.assertEqual(_to_text("纯文本"), "纯文本")
        blocks = [{"type": "text", "text": "第一段"}, {"type": "text", "text": "第二段"}]
        self.assertEqual(_to_text(blocks), "第一段第二段")


if __name__ == "__main__":
    unittest.main()
