"""验证业务查询、协议和真实 HTTP 往返。"""
from datetime import date
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agisample.a2ui.sales_assistant.data import SalesRepository
from agisample.a2ui.sales_assistant.service import SalesService
from agisample.a2ui.sales_assistant.server import create_server

TODAY = date(2026, 9, 22)

def model(messages):
    return next(m["updateDataModel"]["value"] for m in messages if "updateDataModel" in m)

class SalesTests(unittest.TestCase):
    def setUp(self):
        self.service = SalesService(today=TODAY)

    def test_aggregate_matches_details(self):
        with SalesRepository(TODAY) as repo:
            data = repo.snapshot("2026-09", "全部")
            east = repo.snapshot("2026-09", "华东")
            self.assertEqual(data["order_count"], 12)
            self.assertEqual(data["total_cents"], 11587200)
            self.assertEqual(east["total_cents"], 1737600)
            self.assertEqual(data["total_cents"], sum(r["amount_cents"] for r in data["orders"]))
            self.assertEqual(east["order_count"], 3)
            self.assertTrue(all(r["region"] == "华东" for r in east["orders"]))
            self.assertEqual(sum(r["total_cents"] for r in data["regions"]), data["total_cents"])

    def test_month_boundary_and_empty(self):
        self.assertEqual(model(self.service.query("查看上月销售情况"))["filters"]["period"], "2026-08")
        jan = SalesService(today=date(2026, 1, 10))
        self.assertEqual(model(jan.query("上月订单明细"))["filters"]["period"], "2025-12")
        empty = model(self.service.query("查看2020-01销售情况"))
        self.assertEqual(empty["rows"], [])
        self.assertEqual(empty["metrics"]["count"], "0")
        self.assertIn("没有订单", empty["description"])

    def test_protocol_and_action_keep_period(self):
        messages = self.service.query("查看上月各区域销售情况")
        self.assertEqual([next(k for k in m if k != "version") for m in messages],
                         ["createSurface", "updateDataModel", "updateComponents"])
        self.assertTrue(all(m["version"] == "v0.9.1" for m in messages))
        sid = messages[0]["createSurface"]["surfaceId"]
        reply = self.service.action({"version": "v0.9.1", "action": {
            "name": "show_details", "surfaceId": sid, "sourceComponentId": "results",
            "timestamp": "2026-09-22T00:00:00Z", "context": {"period": "2026-08", "region": "华东"}}})
        self.assertNotIn("createSurface", reply[0])
        value = model(reply)
        self.assertEqual(value["filters"], {"period": "2026-08", "region": "华东", "view": "details"})
        self.assertEqual(len(value["rows"]), 3)

    def test_invalid_queries(self):
        for query in ["", None, "a" * 501, "删除订单", "SELECT * FROM orders", "查看2026-13销售",
                      "查看昨天销售", "查看华中销售", "查看利润", "本月和上月销售", "华东华南销售"]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.service.query(query)

    def test_invalid_events(self):
        for payload in [{}, {"action": []}, {"version": "v0.8", "action": {}},
                        {"version": "v0.9.1", "action": {"name": "delete_orders"}}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.service.action(payload)

    def test_query_isolation_and_tree(self):
        first = self.service.query("本月华东订单明细")
        second = self.service.query("本月销售情况")
        self.assertNotEqual(first[0]["createSurface"]["surfaceId"], second[0]["createSurface"]["surfaceId"])
        self.assertEqual(model(second)["filters"]["region"], "全部")
        components = second[-1]["updateComponents"]["components"]
        ids = {c["id"] for c in components}
        self.assertEqual(len(ids), len(components))
        self.assertIn("root", ids)
        for c in components:
            for child in c.get("children", []):
                self.assertIn(child, ids)
            if "child" in c:
                self.assertIn(c["child"], ids)

class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server("127.0.0.1", 0, today=TODAY, mode="demo")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, path, payload):
        return urlopen(Request(self.base + path, json.dumps(payload).encode(),
                       {"Content-Type": "application/json"}), timeout=5)

    def test_http_roundtrip_and_static(self):
        with self.post("/api/query", {"query": "本月销售情况"}) as response:
            self.assertIn("application/x-ndjson", response.headers["Content-Type"])
            messages = [json.loads(line) for line in response.read().decode().splitlines()]
        action = {"version": "v0.9.1", "action": {
            "name": "filter_region", "surfaceId": messages[0]["createSurface"]["surfaceId"],
            "sourceComponentId": "region-1", "timestamp": "2026-09-22T00:00:00Z",
            "context": {"period": "2026-09", "region": "华东", "view": "summary"}}}
        with self.post("/api/action", action) as response:
            reply = [json.loads(line) for line in response.read().decode().splitlines()]
        self.assertEqual(model(reply)["metrics"]["count"], "3")
        for path in ["/", "/app.mjs", "/renderer.mjs", "/state.mjs", "/styles.css", "/catalog.json"]:
            with urlopen(self.base + path, timeout=5) as response:
                self.assertEqual(response.status, 200)


    def test_http_body_errors_are_explicit(self):
        cases = [
            (b"{}", "text/plain", 415),
            (b"x" * 16385, "application/json", 413),
            (b"{", "application/json", 400),
            (bytes([255]), "application/json", 400),
        ]
        for body, content_type, expected in cases:
            request = Request(self.base + "/api/query", body, {"Content-Type": content_type})
            with self.subTest(body=body[:10]), self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=5)
            self.assertEqual(error.exception.code, expected)
            with error.exception as response:
                message = json.loads(response.read())["error"]
            self.assertRegex(message, r"[\u4e00-\u9fff]")

    def test_bad_requests_and_private_files(self):
        for path, payload, status in [("/api/query", [], 400), ("/api/query", {"query": "DROP TABLE orders"}, 400),
                                      ("/api/action", {}, 400), ("/api/unknown", {}, 404)]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                self.post(path, payload)
            self.assertEqual(error.exception.code, status)
        for path in ["/.env", "/../data.py", "/%2e%2e/server.py"]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                urlopen(self.base + path, timeout=5)
            self.assertEqual(error.exception.code, 404)

if __name__ == "__main__":
    unittest.main()
