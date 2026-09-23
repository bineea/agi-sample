"""使用 TypeSafe AI 同时完成客服工单分类、情绪评分和紧急性判断。"""

# 三种问题类型的区别：
# - Choice：回答“属于哪一类”，从无顺序的预设类别中单选；criteria 使用字典。
# - Score：回答“程度有多高”，按照从低到高的等级衡量；criteria 使用有序列表。
# - Noul：回答“是不是”，返回回答“是”的概率；可选 criteria 定义 true/false 的含义。
# 本例分别用它们判断“交给谁”“有多不满”“是否紧急”。
# 官方说明：
# https://docs.typesafe.ai/primitives/choice
# https://docs.typesafe.ai/primitives/score
# https://docs.typesafe.ai/primitives/noul

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
            # Choice 返回概率最高的类别 choice、各类别 probabilities 和 confidence。
            # 类别之间没有高低顺序，概率之和为 1；这是单选分类，不是多标签判断。
            # 若需判断“既涉及退款，又涉及技术故障”，可以拆成两个独立的 Noul 问题。
            "department": Choice(
                instructions="哪个团队最适合处理这条客户工单？",
                criteria={
                    "billing": "账单、扣款、退款或订阅问题",
                    "technical": "软件故障、接口或集成问题",
                    "sales": "价格咨询、购买或售前问题",
                },
            ),
            # Score 的等级编号由列表位置决定，从 0 开始；本例范围为 0～2。
            # score 是等级编号的概率加权平均值，可以为小数，并非只取最可能的等级。
            # 例如等级 0/1/2 的概率为 0.0/0.8/0.2，则 score = 0*0.0 + 1*0.8 + 2*0.2 = 1.2。
            # 同时返回各等级 probabilities、confidence，以及编号到描述的映射 legend。
            # 相同分数可能来自不同分布：全在等级 1，或等级 0 和 2 各占一半，均得到 1.0。
            "frustration": Score(
                instructions="客户在工单中表现出的不满程度有多高？",
                criteria=["情绪平静，只是陈述事实", "感到不满，但表达礼貌", "非常愤怒，措辞强烈"],
            ),
            # Noul 返回 0～1 的“是”的概率，没有独立的 confidence 字段。
            # 接近 1 倾向“是”，接近 0 倾向“否”，接近 0.5 表示判断不明确。
            # 0.9 表示模型认为“存在紧迫性”的概率为 90%，不是紧急程度达到 90 分。
            # 若需衡量紧急程度，应使用 Score 并定义具体等级。
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
        # Choice 和 Score 的 confidence 概括概率分布的集中程度，范围为 0～1。
        # 它不等于最高选项的概率；高置信度也不保证答案正确。
        department = response.choices["department"]
        frustration = response.scores["frustration"]
        urgent = response.nouls["is_urgent"]
        # 如需布尔判断，应显式比较阈值，例如 urgent.noul >= 0.8（仅为示例阈值）。
        # 不要使用 bool(urgent.noul)：即使 bool(0.01) 也为 True。
        # 阈值应按误判成本和实际样本确定，中间的不确定区间可交由人工复核。
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
