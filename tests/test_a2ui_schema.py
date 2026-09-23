"""使用已固定的官方协议 schema 校验本示例消息；仅测试依赖 jsonschema。"""
from datetime import date
import json
from pathlib import Path
import sys
import unittest

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agisample.a2ui.sales_assistant.service import SalesService

class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures = ROOT / "tests/fixtures/a2ui_v0_9_1"
        common = json.loads((fixtures / "common_types.json").read_text(encoding="utf-8"))
        catalog = json.loads((ROOT / "src/agisample/a2ui/sales_assistant/web/catalog.json").read_text(encoding="utf-8-sig"))
        server = json.loads((fixtures / "server_to_client.json").read_text(encoding="utf-8"))
        client = json.loads((fixtures / "client_to_server.json").read_text(encoding="utf-8"))
        resources = [(common["$id"], Resource.from_contents(common)),
                     (catalog["$id"], Resource.from_contents(catalog)),
                     ("https://a2ui.org/specification/v0_9/catalog.json", Resource.from_contents(catalog))]
        registry = Registry().with_resources(resources)
        cls.validator = Draft202012Validator(server, registry=registry)
        cls.client_validator = Draft202012Validator(client, format_checker=FormatChecker())

    def test_generated_queries_and_events_match_official_schema(self):
        service = SalesService(date(2026, 9, 22))
        for query in ("本月销售情况", "上月华东订单明细", "2020-01销售情况"):
            messages = service.query(query)
            for message in messages:
                self.validator.validate(message)
            event = {"version": "v0.9.1", "action": {"name": "show_details",
                "surfaceId": messages[0]["createSurface"]["surfaceId"], "sourceComponentId": "results",
                "timestamp": "2026-09-22T08:00:00Z", "context": {"period": "2026-09", "region": "华东"}}}
            self.client_validator.validate(event)
            for message in service.action(event):
                self.validator.validate(message)

    def test_incomplete_custom_table_fails_schema(self):
        message = {"version": "v0.9.1", "updateComponents": {"surfaceId": "test",
                   "components": [{"id": "root", "component": "DataTable"}]}}
        self.assertTrue(list(self.validator.iter_errors(message)))

if __name__ == "__main__":
    unittest.main()
