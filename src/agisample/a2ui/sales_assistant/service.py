"""规则意图识别和业务编排；后续可在 parse_query 边界接入模型。"""
from datetime import date, datetime
import re
from uuid import uuid4

from .data import REGIONS, SalesRepository, shift_month
from .protocol import build_messages

HELP = "支持查询本月、上月或 YYYY-MM 的销售汇总/订单明细，可指定华东、华南、华北、西部。"


def validate_filters(period, region, view):
    if not isinstance(period, str) or not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", period):
        raise ValueError("月份格式应为 YYYY-MM，例如 2026-09。")
    if not 1900 <= int(period[:4]) <= 9998:
        raise ValueError("年份应在 1900 到 9998 之间。")
    if region not in ("全部", *REGIONS):
        raise ValueError("区域仅支持全部、华东、华南、华北、西部。")
    if view not in ("summary", "details"):
        raise ValueError("视图仅支持销售汇总或订单明细。")


def parse_query(text, today):
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        raise ValueError("请输入 1～500 字的查询。" + HELP)
    text = text.strip()
    periods = re.findall(r"[0-9]{4}-[0-9]{1,2}|本月|上月", text)
    regions = [region for region in REGIONS if region in text]
    if len(periods) > 1 or len(regions) > 1:
        raise ValueError("每次请选择一个月份、一个区域；不填写区域则查询全部。")
    if not any(word in text for word in ("销售", "订单", "明细", "汇总")):
        raise ValueError("暂不支持该查询。" + HELP)
    # 白名单语法让不支持的利润、同比、昨天等问题明确失败，避免返回误导数据。
    remainder = re.sub(r"[0-9]{4}-[0-9]{1,2}", "", text)
    words = ("订单明细", "销售情况", "销售金额", "销售数据", "销售汇总", "销售额", "订单数量",
             "订单数", "各区域", "全部区域", "区域", "全部", "本月", "上月", *REGIONS,
             "请帮我", "帮我", "请", "查看", "查询", "显示", "看看", "统计", "一下", "的",
             "销售", "情况", "汇总", "订单", "明细", "数据")
    for word in words:
        remainder = remainder.replace(word, "")
    if re.sub(r"[\s，。？！?!,.:：]", "", remainder):
        raise ValueError("暂不支持该查询。" + HELP)
    value = periods[0] if periods else "本月"
    period = shift_month(today, -1 if value == "上月" else 0) if value in ("本月", "上月") else value
    region = regions[0] if regions else "全部"
    view = "details" if "明细" in text else "summary"
    validate_filters(period, region, view)
    return period, region, view


class SalesService:
    def __init__(self, today=None):
        # 服务启动日作为固定参考，跨请求保持样例数据稳定。
        self.today = today or date.today()

    def _render(self, period, region, view, surface_id, initial=False):
        validate_filters(period, region, view)
        with SalesRepository(self.today) as repo:
            snapshot = repo.snapshot(period, region)
        return build_messages(snapshot, period, region, view, surface_id, self.today, initial)

    def query(self, text):
        period, region, view = parse_query(text, self.today)
        return self._render(period, region, view, "sales-" + uuid4().hex, initial=True)

    def action(self, message):
        if not isinstance(message, dict) or message.get("version") != "v0.9.1":
            raise ValueError("需要 v0.9.1 的 A2UI action 消息。")
        event = message.get("action")
        if not isinstance(event, dict):
            raise ValueError("action 必须是对象。")
        name = event.get("name")
        if name not in ("filter_region", "show_details", "show_summary"):
            raise ValueError("不支持该操作。")
        surface_id = event.get("surfaceId")
        if not isinstance(surface_id, str) or not re.fullmatch(r"sales-[a-f0-9]{32}", surface_id):
            raise ValueError("无效的查询界面标识，请重新查询。")
        source = event.get("sourceComponentId")
        if not isinstance(source, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", source):
            raise ValueError("缺少有效的事件来源组件。")
        try:
            timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError
        except (KeyError, TypeError, AttributeError, ValueError):
            raise ValueError("事件时间必须是包含时区的 ISO 8601 时间。") from None
        context = event.get("context")
        if not isinstance(context, dict):
            raise ValueError("事件 context 必须是对象。")
        period, region = context.get("period"), context.get("region")
        view = "details" if name == "show_details" else "summary" if name == "show_summary" else context.get("view")
        return self._render(period, region, view, surface_id)
