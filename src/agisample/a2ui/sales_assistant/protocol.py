"""将业务结果转换为 A2UI v0.9.1 消息与项目自定义组件目录。"""
from decimal import Decimal, ROUND_HALF_UP

from .data import REGIONS, shift_month

VERSION = "v0.9.1"
CATALOG_ID = "urn:agisample:a2ui:sales:1"


def money(cents):
    return f"¥{Decimal(cents) / 100:,.2f}"


def build_data(snapshot, period, region, view, today):
    count, total = snapshot["order_count"], snapshot["total_cents"]
    average = int((Decimal(total) / count).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) if count else 0
    rows = []
    if view == "summary":
        for item in snapshot["regions"]:
            rows.append({"region": item["region"], "count": item["order_count"],
                         "amount": money(item["total_cents"]),
                         "share": f'{item["total_cents"] / total:.1%}' if total else "0.0%"})
        columns = [{"key": "region", "label": "区域"}, {"key": "count", "label": "订单数", "align": "right"},
                   {"key": "amount", "label": "销售额", "align": "right"}, {"key": "share", "label": "占比", "align": "right"}]
    else:
        rows = [{**order, "amount": money(order["amount_cents"])} for order in snapshot["orders"]]
        columns = [{"key": "id", "label": "订单编号"}, {"key": "ordered_on", "label": "日期"},
                   {"key": "region", "label": "区域"}, {"key": "customer", "label": "客户"},
                   {"key": "product", "label": "商品"}, {"key": "amount", "label": "金额", "align": "right"}]
    title = f'{period} · {region} · {"销售汇总" if view == "summary" else "订单明细"}'
    description = (f"共 {count} 笔订单，金额为人民币。"
                   if count else "该月份或区域没有订单，请切换月份或区域。")
    data = {"title": title, "description": description,
            "filters": {"period": period, "region": region, "view": view},
            "metrics": {"total": money(total), "count": str(count), "average": money(average)},
            "rows": rows, "sampleNote": f"虚构样例数据覆盖 {shift_month(today, -2)} 至 {shift_month(today, 0)}；参考日期 {today.isoformat()}。"}
    return {**data, "columns": columns}


def build_messages(snapshot, period, region, view, surface_id, today, initial):
    """仅显式 demo 模式使用固定布局；Agent 模式不调用此函数。"""
    data = build_data(snapshot, period, region, view, today)
    columns = data["columns"]
    components = []

    def add(identifier, component, **props):
        components.append({"id": identifier, "component": component, **props})
        return identifier

    def text(identifier, value, variant="body"):
        return add(identifier, "Text", text=value, variant=variant)

    def button(identifier, label, name, context, active=False):
        child = text(identifier + "-label", label)
        return add(identifier, "Button", child=child, variant="primary" if active else "default",
                   action={"event": {"name": name, "context": context}})

    text("heading", {"path": "/title"}, "h2")
    text("description", {"path": "/description"}, "muted")
    metric_ids = []
    for key, label in (("total", "销售额"), ("count", "订单数"), ("average", "平均订单金额")):
        label_id = text(key + "-label", label, "muted")
        value_id = text(key + "-value", {"path": "/metrics/" + key}, "metric")
        content_id = add(key + "-content", "Column", children=[label_id, value_id])
        metric_ids.append(add(key + "-card", "Card", child=content_id))
    add("metrics", "Row", children=metric_ids)
    text("region-label", "按区域筛选", "muted")
    buttons = []
    for index, name in enumerate(("全部", *REGIONS)):
        buttons.append(button(f"region-{index}", name, "filter_region",
                              {"period": {"path": "/filters/period"}, "region": name,
                               "view": {"path": "/filters/view"}}, active=name == region))
    add("region-controls", "Row", children=buttons)
    nav = [button("summary", "区域汇总", "show_summary",
                  {"period": {"path": "/filters/period"}, "region": {"path": "/filters/region"}}, view == "summary"),
           button("details", "订单明细", "show_details",
                  {"period": {"path": "/filters/period"}, "region": {"path": "/filters/region"}}, view == "details")]
    add("view-controls", "Row", children=nav)
    table = {"columns": columns, "rows": {"path": "/rows"}, "emptyText": "暂无订单，试试本月或上月。",
             "caption": {"path": "/title"}}
    if view == "summary":
        table["rowAction"] = {"label": "查看明细", "event": {"name": "show_details",
            "context": {"period": {"path": "/filters/period"}, "region": {"path": "region"}}}}
    add("results", "DataTable", **table)
    text("sample-note", {"path": "/sampleNote"}, "muted")
    add("root", "Column", children=["heading", "description", "metrics", "region-label",
                                    "region-controls", "view-controls", "results", "sample-note"])
    def envelope(kind, **props):
        return {"version": VERSION, kind: {"surfaceId": surface_id, **props}}
    messages = [envelope("createSurface", catalogId=CATALOG_ID)] if initial else []
    messages += [envelope("updateDataModel", path="/", value=data),
                 envelope("updateComponents", components=components)]
    return messages
