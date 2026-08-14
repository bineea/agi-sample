"""把 LangChain v1 create_agent 的 ReAct agent 包装成 A2A AgentExecutor。"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from langchain_core.messages import AIMessage

from agisample.langgraph.a2a.peer_tool import current_hop


def _to_text(content: object) -> str:
    """把 AIMessage 的 content（str 或 content-block 列表）归一化为纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


class LangChainAgentExecutor(AgentExecutor):
    """每个请求：读 hop → 注入 contextvar → ainvoke → 回写纯文本 Message。"""

    def __init__(self, agent: object) -> None:
        # create_agent 返回的 CompiledStateGraph
        self._agent = agent

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        query = context.get_user_input()
        # 从 A2A 消息 metadata 读对端传来的 hop（client 首发时缺省为 0）
        hop = 0
        if context.message and context.message.metadata:
            hop = int(context.message.metadata.get("hop", 0))

        token = current_hop.set(hop)
        try:
            # 把 hop 显式写进用户输入，agent 据此决定回调时传 hop=N+1
            prompt = f"[A2A hop={hop}] {query}"
            result = await self._agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                reply = _to_text(last_message.content)
            else:
                reply = str(last_message.content)
        except Exception as exc:  # LLM 异常：不拖垮 server，返回错误文本
            reply = f"处理时发生错误：{type(exc).__name__}: {exc}"
        finally:
            current_hop.reset(token)

        # 纯 Message 响应（非 Task 模式），客户端同步拿到回复
        await event_queue.enqueue_event(new_agent_text_message(reply))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("本 demo 不支持任务取消")
