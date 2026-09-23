"""虚构订单与参数化只读 SQLite 查询；金额以整数分保存。"""
from datetime import date
import sqlite3

REGIONS = ("华东", "华南", "华北", "西部")


def shift_month(day: date, offset: int) -> str:
    index = day.year * 12 + day.month - 1 + offset
    year, month = divmod(index, 12)
    return f"{year:04d}-{month + 1:02d}"


class SalesRepository:
    def __init__(self, today: date):
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(
            "CREATE TABLE orders (id TEXT PRIMARY KEY, ordered_on TEXT, region TEXT, "
            "customer TEXT, product TEXT, amount_cents INTEGER NOT NULL)")
        rows = []
        products = ("商务笔记本", "协作显示器", "桌面工作站")
        for offset in (-2, -1, 0):
            period = shift_month(today, offset)
            for region_index, region in enumerate(REGIONS):
                for item in range(3):
                    day = min(3 + item * 7, today.day) if offset == 0 else 3 + item * 7
                    rows.append((
                        f"SO-{period.replace('-', '')}-{region_index + 1}{item + 1}",
                        f"{period}-{day:02d}", region, f"{region}示例客户{item + 1}",
                        products[item], (region_index + 2) * (item + 1) * 128800 + (offset + 2) * 32000))
        self.connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", rows)
        self.connection.commit()
        self.connection.execute("PRAGMA query_only = ON")

    def snapshot(self, period: str, region: str) -> dict:
        # 不执行用户 SQL；月份以闭开区间过滤，避免跨月数据混入。
        year, month = map(int, period.split("-"))
        start = date(year, month, 1).isoformat()
        end = shift_month(date(year, month, 1), 1) + "-01"
        where = "ordered_on >= ? AND ordered_on < ?"
        params = [start, end]
        if region != "全部":
            where += " AND region = ?"
            params.append(region)
        orders = [dict(row) for row in self.connection.execute(
            f"SELECT * FROM orders WHERE {where} ORDER BY ordered_on DESC, id", params)]
        regions = [dict(row) for row in self.connection.execute(
            f"SELECT region, COUNT(*) AS order_count, SUM(amount_cents) AS total_cents "
            f"FROM orders WHERE {where} GROUP BY region ORDER BY total_cents DESC, region", params)]
        return {"orders": orders, "regions": regions, "order_count": len(orders),
                "total_cents": sum(order["amount_cents"] for order in orders)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()
