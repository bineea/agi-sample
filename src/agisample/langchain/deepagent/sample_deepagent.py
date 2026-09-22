import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend

# 需要在执行前由用户确认的工具（对应 agentscope 示例里的 RequireUserConfirmEvent）
# LocalShellBackend 提供文件系统工具（read_file/write_file/edit_file/ls/glob/grep）
# 以及 shell 执行工具 execute。
INTERRUPT_ON = {
    "write_file": True,
    "edit_file": True,
    "execute": True,
}

SYSTEM_PROMPT = (
    "You are a helpful assistant named Friday. "
    "You can answer questions and use tools when necessary. "
    "When using file or shell tools, explain what you are doing."
)


def build_agent():
    load_dotenv(find_dotenv())

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 API_KEY")

    model = ChatOpenAI(
        model=os.getenv("MODEL", ""),
        api_key=api_key,
        base_url=os.getenv("BASE_URL") or None,
        temperature=0,
    )

    # 本地后端：同时提供文件系统工具和 execute(shell) 能力。
    # 仅在受控的开发环境使用，execute 等价于在本机直接执行命令。
    backend = LocalShellBackend(root_dir=".", virtual_mode=True)

    # checkpointer 用于在人工确认（interrupt）之后从断点恢复执行。
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=SYSTEM_PROMPT,
        interrupt_on=INTERRUPT_ON,
        checkpointer=MemorySaver(),
    )


def _collect_decisions(interrupts) -> list[dict]:
    """
    针对每个待确认的工具调用询问用户，返回 decisions 列表。
    """
    decisions = []
    for interrupt in interrupts:
        hitl_request = interrupt.value
        for action in hitl_request["action_requests"]:
            print(f"\n工具: {action['name']}")
            print(f"参数: {action['args']}")

            answer = input("是否允许执行？[y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                decisions.append({"type": "approve"})
            else:
                decisions.append(
                    {
                        "type": "reject",
                        "message": "用户拒绝了该工具调用，请不要重试，除非用户再次要求。",
                    }
                )
    return decisions


def stream_reply(agent, config: dict, user_input: str) -> None:
    """
    发送一条用户消息，并流式打印 Friday 的回复；遇到需要确认的工具调用时暂停并询问。
    """
    current_input = {"messages": [{"role": "user", "content": user_input}]}

    while True:
        pending_interrupts = []

        for chunk in agent.stream(
            current_input,
            config=config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, _metadata = chunk["data"]
                if token.content:
                    print(token.content, end="", flush=True)

            elif chunk["type"] == "updates":
                for node_name, update in chunk["data"].items():
                    if node_name == "__interrupt__":
                        pending_interrupts.extend(update)
                    elif node_name == "tools":
                        print(f"\n\n[Friday 正在调用工具]\n", flush=True)

        if not pending_interrupts:
            break

        # 有需要确认的工具调用：收集用户决定并从断点恢复
        print("\n[需要你确认以下工具调用]\n")
        decisions = _collect_decisions(pending_interrupts)
        current_input = Command(resume={"decisions": decisions})

    print()


def main() -> None:
    agent = build_agent()
    # thread_id 让 checkpointer 能定位到本次会话的状态，实现 interrupt 恢复与多轮对话
    config = {"configurable": {"thread_id": "friday-session"}}

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
            stream_reply(agent, config, user_input)
        except Exception as exc:
            print(f"\n[发生错误] {type(exc).__name__}: {exc}")


# Offload Context
# deepagents 通过文件系统后端把超长上下文（被压缩/截断的内容）卸载到文件中，
# agent 之后可以用 read_file/grep/glob 等工具回查这些被移除的细节。

if __name__ == "__main__":
    main()
