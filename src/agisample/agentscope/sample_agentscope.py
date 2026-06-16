import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.event import (
    TextBlockDeltaEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ReplyEndEvent,
    RequireUserConfirmEvent,
    ConfirmResult,
    UserConfirmResultEvent,
)
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, Bash, Read, Write, Edit


def build_agent() -> Agent:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 API_KEY")

    return Agent(
        name="Friday",
        system_prompt=(
            "You are a helpful assistant named Friday. "
            "You can answer questions and use tools when necessary. "
            "When using file or shell tools, explain what you are doing."
        ),
        model=OpenAIChatModel(
            credential=OpenAICredential(
                api_key=api_key,
                # base_url=""
            ),
            model="",
        ),
        toolkit=Toolkit(
            tools=[
                Bash(),
                Read(),
                Write(),
                Edit(),
            ]
        ),
    )


async def stream_reply(agent: Agent, user_input: str) -> None:
    """
    发送一条用户消息，并流式打印 Friday 的回复。
    """
    current_input = UserMsg(name="user", content=user_input)

    while True:
        need_resume = False

        async for event in agent.reply_stream(current_input):
            if isinstance(event, TextBlockDeltaEvent):
                print(event.delta, end="", flush=True)

            elif isinstance(event, ToolCallStartEvent):
                print(f"\n\n[Friday 正在调用工具: {event.tool_call_name}]\n", flush=True)

            elif isinstance(event, ToolResultEndEvent):
                print(f"\n[工具执行完成: {event.state}]\n", flush=True)

            elif isinstance(event, RequireUserConfirmEvent):
                print("\n[需要你确认以下工具调用]\n")

                confirm_results = []

                for tool_call in event.tool_calls:
                    print(f"工具: {tool_call.name}")
                    print(f"参数: {tool_call.input}")

                    answer = input("是否允许执行？[y/N]: ").strip().lower()
                    confirmed = answer in {"y", "yes"}

                    confirm_results.append(
                        ConfirmResult(
                            confirmed=confirmed,
                            tool_call=tool_call,
                            # 接受 suggested_rules 后，类似调用后续可能不再反复询问
                            rules=tool_call.suggested_rules if confirmed else [],
                        )
                    )

                current_input = UserConfirmResultEvent(
                    reply_id=event.reply_id,
                    confirm_results=confirm_results,
                )

                need_resume = True
                break

            elif isinstance(event, ReplyEndEvent):
                print()

        if not need_resume:
            break


async def main() -> None:
    agent = build_agent()

    print("Friday 已启动。输入 /exit 或 /quit 退出。")
    print("你可以问问题，也可以让它读写文件或执行命令。\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            print("退出。")
            break

        print("Friday: ", end="", flush=True)

        try:
            await stream_reply(agent, user_input)
        except Exception as exc:
            print(f"\n[发生错误] {type(exc).__name__}: {exc}")


# Offload Context
# Context offload 是把 agent 已经从上下文中移除的内容 —— 被压缩的消息、被截断的工具输出 —— 写入外部存储，方便 agent 之后通过文件工具（Read、Grep、Glob）回查那些被压缩走的细节。

if __name__ == "__main__":
    asyncio.run(main())