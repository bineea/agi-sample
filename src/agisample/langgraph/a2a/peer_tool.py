"""把"通过 A2A 协议询问对方 agent"封装成 LangChain 工具。

hop 计数用于防止 A↔B 双向互调无限循环，经三条通道传递与兜底：
1. 用户输入中的 [A2A hop=N] 前缀，提示 LLM 调用工具时传 hop=N+1；
2. A2A Message.metadata 中的 hop 字段，跨进程传给对端 server；
3. 本模块的 current_hop ContextVar，由对端 executor 写入，工具内取
   max(LLM 传入值, contextvar + 1) 兜底校正，超过 MAX_HOP 直接短路。
"""

import uuid
from contextvars import ContextVar

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.client.errors import A2AClientHTTPError
from a2a.types import (
    JSONRPCErrorResponse,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    Task,
    TextPart,
)
from langchain_core.tools import BaseTool, tool

# 单个请求允许的最大 agent 间互调次数（client → A 不计，A→B 为 1，B→A 为 2）
MAX_HOP = 2

# 调用对方服务的超时时间（秒），对方内部可能也是一次完整的 LLM 调用
REQUEST_TIMEOUT = 120.0

# 当前请求在本进程内的 hop 计数，由 server 端 executor 在调用 agent 前写入
current_hop: ContextVar[int] = ContextVar("a2a_current_hop", default=0)


def _extract_reply_text(result: Message | Task) -> str:
    """从 A2A 响应结果（Message 或 Task）中提取纯文本。"""
    if isinstance(result, Message):
        return "".join(
            part.root.text for part in result.parts if isinstance(part.root, TextPart)
        )
    # Task：拼接所有 artifact 中的文本
    texts: list[str] = []
    for artifact in result.artifacts or []:
        texts.extend(
            part.root.text for part in artifact.parts if isinstance(part.root, TextPart)
        )
    return "".join(texts)


async def ask_peer_text(question: str, peer_url: str, hop: int) -> str:
    """通过 A2A 协议向指定地址的 agent 发送问题并返回其文本回复。

    任何失败（对方未启动、超时、对端报错）都转成中文提示字符串返回，
    不向调用方抛异常。

    :param question: 要问对方的完整问题
    :param peer_url: 对方 A2A 服务地址（如 "http://localhost:10002"）
    :param hop: 本次互调的 hop 计数，随消息 metadata 传给对端
    """
    message = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[Part(root=TextPart(text=question))],
        metadata={"hop": hop},
    )
    request = SendMessageRequest(
        id=str(uuid.uuid4()), params=MessageSendParams(message=message)
    )

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as httpx_client:
            # 每次调用都重新解析 agent card，天然成为"对方是否在线"的探活
            card = await A2ACardResolver(
                httpx_client, base_url=peer_url
            ).get_agent_card()
            client = A2AClient(httpx_client=httpx_client, agent_card=card)
            response = await client.send_message(request)
    except (httpx.ConnectError, A2AClientHTTPError):
        # A2ACardResolver 会把底层连接失败包装成 A2AClientHTTPError
        return f"无法连接对方服务（{peer_url}），对方可能未启动。"
    except httpx.TimeoutException:
        return f"调用对方服务（{peer_url}）超时。"
    except Exception as exc:  # 兜底：任何异常都转成文本，由 agent 降级处理
        return f"调用对方服务（{peer_url}）出错：{type(exc).__name__}: {exc}"

    root = response.root
    if isinstance(root, JSONRPCErrorResponse):
        return f"对方服务返回错误：{root.error.message}"
    return _extract_reply_text(root.result)


def make_ask_peer_tool(peer_name: str, peer_url: str) -> BaseTool:
    """生成一个"询问对方 agent"的异步工具。

    :param peer_name: 对方名称（如 "AgentB"），用于工具名与提示文案
    :param peer_url: 对方 A2A 服务地址（如 "http://localhost:10002"）
    """

    @tool(f"ask_{peer_name.lower()}")
    async def ask_peer(question: str, hop: int) -> str:
        """通过 A2A 协议向 {peer_name} 提问并等待其回复。

        :param question: 要问对方的完整问题
        :param hop: 本次互调的 hop 计数，必须传入「当前 hop + 1」
        """
        # 兜底：以本进程 contextvar 为准，防止 LLM 传错 hop 绕过上限
        effective_hop = max(hop, current_hop.get() + 1)
        if effective_hop > MAX_HOP:
            return (
                f"已达到最大互调次数上限（{MAX_HOP}），不能再调用 {peer_name}。"
                "请基于已获得的信息直接回答。"
            )

        reply = await ask_peer_text(question, peer_url, effective_hop)
        # ask_peer_text 内部已把各类失败转成中文提示，这里统一加上来源前缀
        return f"[来自 {peer_name}] {reply}"

    # 工厂模式下 @tool 的 docstring 无法做 f-string 插值，手动补全对方名称
    ask_peer.description = ask_peer.description.replace("{peer_name}", peer_name)
    return ask_peer
