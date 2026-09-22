"""使用 TypeSafe AI 同时完成客服工单分类、情绪评分和紧急性判断。"""

import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from typesafe_sdk import (
    Choice, Noul, Score, SystemOneResponse, TypeSafeClient,
    TypeSafeAPIError, TypeSafeError,
)


DEFAULT_TICKET = "支付接口已经连续三天连接失败，导致我们无法收款、不断损失订单，请尽快帮忙解决！"


def analyze_ticket(client: TypeSafeClient, ticket: str, model: str) -> SystemOneResponse:
    """在一次请求中完成三个维度的分析。"""
    return client.system_one(
        state=ticket,
        model=model,
        questions={
            "department": Choice(
                instructions="哪个团队最适合处理这条客户工单？",
                criteria={
                    "billing": "账单、扣款、退款或订阅问题",
                    "technical": "软件故障、接口或集成问题",
                    "sales": "价格咨询、购买或售前问题",
                },
            ),
            "frustration": Score(
                instructions="客户在工单中表现出的不满程度有多高？",
                criteria=["情绪平静，只是陈述事实", "感到不满，但表达礼貌", "非常愤怒，措辞强烈"],
            ),
            "is_urgent": Noul(instructions="这条工单是否表达了紧迫性或时间敏感性？"),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TICKET, help="待分析的工单文本（默认使用内置中文示例）")
    parser.add_argument("--model", default="jev-latest", help="模型名称（默认：jev-latest）")
    args = parser.parse_args(argv)
    if not args.text.strip() or not args.model.strip():
        print("错误：工单文本和模型名称不能为空。", file=sys.stderr)
        return 2

    # 固定从项目根目录加载，避免运行目录影响配置；不覆盖已有环境变量。
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    if not os.getenv("TYPESAFE_API_KEY", "").strip():
        print("错误：请设置环境变量 TYPESAFE_API_KEY，或在项目根目录 .env 中配置。", file=sys.stderr)
        return 2

    try:
        with TypeSafeClient(timeout=30.0) as client:
            response = analyze_ticket(client, args.text, args.model)
        # 使用 SDK 的类型化访问器，同时检查响应是否包含所需结果。
        department = response.choices["department"]
        frustration = response.scores["frustration"]
        urgent = response.nouls["is_urgent"]
    except TypeSafeAPIError as exc:
        print(f"错误：TypeSafe API 调用失败（HTTP {exc.status}），请检查密钥、权限、额度和模型名称。", file=sys.stderr)
        return 1
    except TypeSafeError:
        print("错误：TypeSafe 请求失败，请检查配置、网络连接及服务响应。", file=sys.stderr)
        return 1
    except KeyError:
        print("错误：服务响应缺少预期的分类、评分或紧急性结果。", file=sys.stderr)
        return 1

    print(f"模型：{response.model}")
    print(f"处理部门：{department.choice}（置信度：{department.confidence}）")
    print(f"不满程度：{frustration.score}（0=平静，1=不满但礼貌，2=非常愤怒；可能为小数）")
    print(f"紧急性：{urgent.noul}（0～1，越高越支持存在紧迫性）")
    print("完整响应：")
    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
