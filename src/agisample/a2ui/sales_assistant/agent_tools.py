"""受限业务工具：只接受结构化过滤参数，不执行模型生成的 SQL。"""
from .data import REGIONS, SalesRepository
from .protocol import build_data
from .service import validate_filters

QUERY_TOOL = {"type": "function", "function": {
    "name": "query_sales",
    "description": "只读查询虚构销售数据。返回准确金额、订单数、表格行、列定义及数据绑定路径信息。每次只查一个月份和一个区域；可多次调用比较。",
    "parameters": {"type": "object", "properties": {
        "period": {"type": "string", "pattern": "^[0-9]{4}-(0[1-9]|1[0-2])$", "description": "年月 YYYY-MM"},
        "region": {"type": "string", "enum": ["全部", *REGIONS]},
        "view": {"type": "string", "enum": ["summary", "details"]}},
        "required": ["period", "region", "view"], "additionalProperties": False}}}


def run_query(arguments, today):
    if not isinstance(arguments, dict) or set(arguments) != {"period", "region", "view"}:
        raise ValueError("query_sales 仅接受 period、region、view，禁止 SQL 或其他参数。")
    period, region, view = (arguments[key] for key in ("period", "region", "view"))
    validate_filters(period, region, view)
    with SalesRepository(today) as repo:
        snapshot = repo.snapshot(period, region)
    return build_data(snapshot, period, region, view, today)
