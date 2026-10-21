"""受限业务工具：只接受结构化过滤参数，不执行模型生成的 SQL。

与 a2ui 版不同：本示例中模型不通过服务端 tool_calls 取数，而是在 openui-lang
中编写 Query("query_sales", ...) 语句，由前端 toolProvider 回调本模块。
"""
from datetime import date
import re

from .data import REGIONS, SalesRepository, format_yuan

TOOL_NAME = "query_sales"


def validate_filters(period, region, view):
    if not isinstance(period, str) or not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", period):
        raise ValueError("月份格式应为 YYYY-MM，例如 2026-09。")
    if not 1900 <= int(period[:4]) <= 9998:
        raise ValueError("年份应在 1900 到 9998 之间。")
    if region not in ("全部", *REGIONS):
        raise ValueError("区域仅支持全部、华东、华南、华北、西部。")
    if view not in ("summary", "details"):
        raise ValueError("视图仅支持 summary 或 details。")


def run_query(arguments, today: date) -> dict:
    """执行 query_sales 并返回可 JSON 序列化的结果；金额在服务端格式化。"""
    if not isinstance(arguments, dict) or set(arguments) != {"period", "region", "view"}:
        raise ValueError("query_sales 仅接受 period、region、view 三个参数。")
    period, region, view = (arguments[key] for key in ("period", "region", "view"))
    validate_filters(period, region, view)
    with SalesRepository(today) as repo:
        snapshot = repo.snapshot(period, region)
    total_cents = snapshot["total_cents"]
    count = snapshot["order_count"]
    orders = [
        {
            "id": order["id"],
            "ordered_on": order["ordered_on"],
            "region": order["region"],
            "customer": order["customer"],
            "product": order["product"],
            "amount": format_yuan(order["amount_cents"]),
            "amount_cents": order["amount_cents"],
        }
        for order in snapshot["orders"]
    ]
    regions = [
        {
            "region": row["region"],
            "order_count": row["order_count"],
            "total": format_yuan(row["total_cents"]),
            "total_cents": row["total_cents"],
            "total_value": row["total_cents"] / 100,
        }
        for row in snapshot["regions"]
    ]
    view_label = "销售汇总" if view == "summary" else "订单明细"
    return {
        "title": f"{period} {region} {view_label}",
        "filters": {"period": period, "region": region, "view": view},
        "metrics": {
            "total": format_yuan(total_cents),
            "count": count,
            "avg": format_yuan(round(total_cents / count)) if count else "¥0.00",
        },
        "regions": regions,
        "orders": orders,
    }
