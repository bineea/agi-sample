"""离线演示模型：不调用大模型，按关键词返回预置 openui-lang 程序。

只用于无密钥学习和链路验证；Query 取数仍走真实后端工具与 SQLite。
默认 agent 模式不会自动降级到本模块。
"""
from datetime import date

DASHBOARD = """$region = "全部"
sales = Query("query_sales", {period: "__PERIOD__", region: $region, view: "summary"}, {title: "加载中", metrics: {total: "…", count: 0, avg: "…"}, regions: []})
root = Stack([head, picker, stats, chart, actions], "column", "l")
head = TextContent(sales.title)
picker = FormControl("区域（切换自动重新查询，不经过模型）", Select("region", [SelectItem("全部"), SelectItem("华东"), SelectItem("华南"), SelectItem("华北"), SelectItem("西部")], null, $region))
stats = Stack([k1, k2, k3], "row", "m")
k1 = StatCard("销售额", sales.metrics.total)
k2 = StatCard("订单数", "" + sales.metrics.count)
k3 = StatCard("平均订单金额", sales.metrics.avg)
chart = BarChart("区域销售额（元）", sales.regions.region, sales.regions.total_value)
actions = Buttons([btn])
btn = Button("查看订单明细", Action([@ToAssistant("查看本月订单明细")]), "primary")"""

DETAILS = """details = Query("query_sales", {period: "__PERIOD__", region: "全部", view: "details"}, {title: "加载中", orders: []})
root = Stack([tbl, actions], "column", "m")
tbl = DataTable(details.title, [{key: "id", label: "订单号"}, {key: "ordered_on", label: "日期"}, {key: "region", label: "区域"}, {key: "customer", label: "客户"}, {key: "product", label: "产品"}, {key: "amount", label: "金额", align: "right"}], details.orders, "暂无订单")
actions = Buttons([backBtn])
backBtn = Button("返回销售汇总", Action([@ToAssistant("看看本月销售汇总")]), "secondary")"""

HELP = """root = Stack([head, hint], "column", "m")
head = TextContent("离线演示模式：未连接大模型，内置两个固定响应。")
hint = TextContent("试试：看看这个月各个区域的销售表现 / 查看本月订单明细。")"""


class DemoModel:
    """与 ChatModel 相同的流式接口；按是否含"明细"选择预置程序。"""

    model = "demo-offline"

    def complete_stream(self, messages, timeout):
        last = messages[-1]["content"] if messages else ""
        period = date.today().strftime("%Y-%m")
        if "明细" in last:
            program = DETAILS
        elif any(word in last for word in ("销售", "订单", "汇总", "对比")):
            program = DASHBOARD
        else:
            program = HELP
        program = program.replace("__PERIOD__", period)
        for i in range(0, len(program), 24):
            yield program[i:i + 24]

    def close(self):
        pass
