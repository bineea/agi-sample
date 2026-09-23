"""真实 SDK 使用模拟 HTTP；验证 Agent 工具循环和模型生成的界面，不花费 API 额度。"""
import copy
from datetime import date
import json
from pathlib import Path
import sys
import os
import tempfile
import threading
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import unittest

import httpx
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agisample.a2ui.sales_assistant.agent import AgentService, AgentError
from agisample.a2ui.sales_assistant.llm import ChatModel, ModelConfig
from agisample.a2ui.sales_assistant.agent_tools import run_query

TODAY = date(2026, 9, 22)

def call(period="2026-09", region="全部", view="summary", name="query_sales", extra=None):
    args = {"period": period, "region": region, "view": view, **(extra or {})}
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "tool-1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

def final(reply="已查询，点击表格可以查看明细。", invalid=False):
    components = [
        {"id": "root", "component": "Column", "children": ["heading", "amount", "table"]},
        {"id": "heading", "component": "Text", "text": "模型选择的销售视图", "variant": "h2"},
        {"id": "amount", "component": "Text", "text": {"path": "/queries/q1/metrics/total"}, "variant": "metric"},
        {"id": "table", "component": "DataTable", "caption": {"path": "/queries/q1/title"},
         "rows": {"path": "/queries/q1/rows"}, "columns": [{"key": "region", "label": "区域"}],
         "emptyText": "暂无订单", "rowAction": {"label": "查看明细", "event": {
             "name": "show_details", "context": {"period": {"path": "/queries/q1/filters/period"}, "region": {"path": "region"}}}}},
    ]
    if invalid:
        components[2]["text"] = "¥999999"
    return {"role": "assistant", "content": json.dumps({"reply": reply, "components": components})}

class ScriptedModel:
    model = "mock-model"
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
    def complete(self, messages, tools, timeout):
        self.requests.append(copy.deepcopy(messages))
        item = next(self.replies)
        if isinstance(item, Exception):
            raise item
        return copy.deepcopy(item)

def value(result):
    return next(m["updateDataModel"]["value"] for m in result.messages if "updateDataModel" in m)

def click(result, region="华东"):
    sid = next(m["createSurface"]["surfaceId"] for m in result.messages if "createSurface" in m)
    return {"version": "v0.9.1", "action": {
        "name": "show_details", "surfaceId": sid, "sourceComponentId": "table",
        "timestamp": "2026-09-22T08:00:00Z", "context": {"period": "2026-09", "region": region}}}

class AgentTests(unittest.TestCase):
    def test_tool_loop_and_model_layout(self):
        model = ScriptedModel([call(), final()])
        service = AgentService(model, TODAY)
        result = service.query("看看这个月的销售表现")
        data = value(result)
        self.assertEqual(data["queries"]["q1"]["metrics"]["total"], "¥115,872.00")
        self.assertEqual(data["agent"]["reply"], "已查询，点击表格可以查看明细。")
        comps = result.messages[-1]["updateComponents"]["components"]
        self.assertEqual(comps[1]["text"], "模型选择的销售视图")
        self.assertEqual(len(comps), 4)  # 不是原来的固定模板
        tool = model.requests[1][-1]
        self.assertEqual(tool["role"], "tool")
        self.assertEqual(tool["tool_call_id"], "tool-1")
        self.assertEqual(json.loads(tool["content"])["queryId"], "q1")

    def test_followup_and_click_use_same_agent(self):
        model = ScriptedModel([call(), final(), call(region="华东", view="details"), final(),
                               call(period="2026-08", region="华东"), final()])
        service = AgentService(model, TODAY)
        first = service.query("本月销售")
        second = service.action(click(first), first.session_id)
        self.assertEqual(second.session_id, first.session_id)
        self.assertEqual(value(second)["queries"]["q1"]["filters"]["region"], "华东")
        self.assertIn("show_details", model.requests[2][-1]["content"])
        third = service.query("那上个月呢？", first.session_id)
        self.assertEqual(value(third)["queries"]["q1"]["filters"]["period"], "2026-08")
        self.assertTrue(any(m.get("content") == "本月销售" for m in model.requests[4]))
        self.assertTrue(any("show_details" in (m.get("content") or "") for m in model.requests[4] if m["role"] == "user"))

    def test_invalid_layout_is_repaired_once(self):
        model = ScriptedModel([call(), final(invalid=True), final()])
        result = AgentService(model, TODAY).query("本月销售")
        self.assertEqual(len(model.requests), 3)
        self.assertIn("校验", model.requests[2][-1]["content"])
        self.assertEqual(value(result)["queries"]["q1"]["metrics"]["count"], "12")

    def test_failed_turn_does_not_commit_and_no_template_fallback(self):
        model = ScriptedModel([call(), final(), call(), final(invalid=True), final(invalid=True)])
        service = AgentService(model, TODAY)
        first = service.query("本月销售")
        before = copy.deepcopy(service.sessions[first.session_id].history)
        with self.assertRaises(AgentError):
            service.query("继续", first.session_id)
        self.assertEqual(service.sessions[first.session_id].history, before)

    def test_foreign_session_and_forged_actions_rejected_without_model_call(self):
        model = ScriptedModel([call(), final(), call(), final()])
        service = AgentService(model, TODAY)
        first, other = service.query("本月销售"), service.query("本月销售")
        with self.assertRaises(ValueError):
            service.action(click(first), other.session_id)
        forged = click(first)
        forged["action"]["context"]["period"] = "2026-08"
        with self.assertRaises(ValueError):
            service.action(forged, first.session_id)
        self.assertEqual(len(model.requests), 4)
        self.assertNotEqual(first.session_id, other.session_id)

    def test_tools_reject_sql_and_bad_filters(self):
        for args in [{"period": "2026-09", "region": "全部", "view": "summary", "sql": "DROP TABLE orders"},
                     {"period": "2026-13", "region": "全部", "view": "summary"},
                     {"period": "2026-09", "region": "华中", "view": "summary"}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                run_query(args, TODAY)

    def test_unknown_tool_returns_error_to_model(self):
        model = ScriptedModel([call(name="execute_sql"), call(), final()])
        result = AgentService(model, TODAY).query("本月销售")
        self.assertIn("error", json.loads(model.requests[1][-1]["content"]))
        self.assertEqual(value(result)["queries"]["q1"]["metrics"]["count"], "12")

    def test_tool_loop_has_limit(self):
        model = ScriptedModel([call()] * 10)
        with self.assertRaises(AgentError):
            AgentService(model, TODAY).query("本月销售")
        self.assertLessEqual(len(model.requests), 6)


    def test_config_aliases_blank_values_and_no_secret_repr(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            env = Path(folder) / ".env"
            env.write_text("OPENAI_MODEL\nAPI_KEY=private-key\nMODEL=test-model\nBASE_URL=https://unit.test/v1\n", encoding="utf-8")
            config = ModelConfig.from_env(env)
            self.assertEqual(config.model, "test-model")
            self.assertNotIn("private-key", repr(config))
            with patch.dict(os.environ, {"MODEL": "override-model"}):
                self.assertEqual(ModelConfig.from_env(env).model, "override-model")

    def test_shared_component_graph_is_rejected(self):
        output = final()
        layout = json.loads(output["content"])
        layout["components"][0]["children"].append("amount")
        output["content"] = json.dumps(layout)
        service = AgentService(ScriptedModel([call(), output, output]), TODAY)
        with self.assertRaises(AgentError):
            service.query("本月销售")

    def test_session_busy_expired_and_turn_limit(self):
        service = AgentService(ScriptedModel([call(), final()]), TODAY)
        result = service.query("本月销售")
        session = service.sessions[result.session_id]
        session.lock.acquire()
        try:
            with self.assertRaises(AgentError) as error:
                service.query("继续", result.session_id)
            self.assertEqual(error.exception.status, 409)
        finally:
            session.lock.release()
        session.turns = 12
        with self.assertRaises(AgentError):
            service.query("继续", result.session_id)
        session.touched = 0
        with self.assertRaises(AgentError):
            service.query("继续", result.session_id)
        self.assertNotIn(result.session_id, service.sessions)

    def test_model_api_error_is_sanitized(self):
        def handle(request):
            return httpx.Response(401, json={"error": {"message": "private-secret-in-provider-error"}})
        sdk = OpenAI(api_key="private-secret", base_url="https://unit.test/v1", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        try:
            client = ChatModel(ModelConfig("private-secret", "fake-model"), client=sdk)
            with self.assertRaises(AgentError) as error:
                AgentService(client, TODAY).query("本月销售")
            self.assertNotIn("private-secret", str(error.exception))
        finally:
            sdk.close()


    def test_multiple_queries_generate_separate_bound_metrics(self):
        layout = {"reply": "已并列展示两个查询范围。", "components": [
            {"id": "root", "component": "Row", "children": ["current", "previous"]},
            {"id": "current", "component": "Text", "text": {"path": "/queries/q1/metrics/total"}, "variant": "metric"},
            {"id": "previous", "component": "Text", "text": {"path": "/queries/q2/metrics/total"}, "variant": "metric"}]}
        model = ScriptedModel([call(), call(period="2026-08"),
                               {"role": "assistant", "content": json.dumps(layout)}])
        result = AgentService(model, TODAY).query("本月和上月并列看")
        queries = value(result)["queries"]
        self.assertEqual(set(queries), {"q1", "q2"})
        self.assertNotEqual(queries["q1"]["metrics"]["total"], queries["q2"]["metrics"]["total"])

    def test_clarification_does_not_fabricate_query_data(self):
        layout = {"reply": "请说明想查询的月份和区域。", "components": [
            {"id": "root", "component": "Text", "text": "你想查看哪个月份？"}]}
        model = ScriptedModel([{"role": "assistant", "content": json.dumps(layout)}])
        result = AgentService(model, TODAY).query("帮我查一下")
        self.assertEqual(value(result)["queries"], {})
        self.assertEqual(value(result)["agent"]["steps"], [])

    def test_sdk_chat_completions_contract(self):
        requests = []
        replies = iter([call(), final()])
        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer fake-test-key")
            return httpx.Response(200, json={"id": "completion-test", "object": "chat.completion", "created": 1,
                "model": "fake-model", "choices": [{"index": 0, "message": next(replies), "finish_reason": "stop"}]})
        sdk = OpenAI(api_key="fake-test-key", base_url="https://unit.test/v1", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        try:
            client = ChatModel(ModelConfig("fake-test-key", "fake-model", "https://unit.test/v1"), client=sdk)
            AgentService(client, TODAY).query("本月销售")
        finally:
            sdk.close()
        self.assertEqual(requests[0]["tools"][0]["function"]["name"], "query_sales")
        self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})




class AgentHTTPTests(unittest.TestCase):
    def test_real_http_session_followup_and_action(self):
        from agisample.a2ui.sales_assistant.server import create_server
        model = ScriptedModel([call(), final(), call(period="2026-08"), final(),
                               call(period="2026-08", region="华东", view="details"), final()])
        service = AgentService(model, TODAY)
        server = create_server("127.0.0.1", 0, service=service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        def post(path, body, session=None):
            headers = {"Content-Type": "application/json"}
            if session:
                headers["X-A2UI-Session"] = session
            with urlopen(Request(base + path, json.dumps(body).encode(), headers), timeout=10) as response:
                return response.headers.get("X-A2UI-Session"), [json.loads(line) for line in response.read().splitlines()]
        try:
            with urlopen(base + "/api/info", timeout=5) as response:
                info = json.load(response)
                self.assertEqual(info["mode"], "agent")
                self.assertEqual(info["model"], "mock-model")
                self.assertNotIn("api_key", info)
            token, first = post("/api/query", {"query": "本月销售"})
            again, second = post("/api/query", {"query": "那上个月呢？"}, token)
            self.assertEqual(token, again)
            self.assertIn("deleteSurface", second[0])
            surface = next(m["createSurface"]["surfaceId"] for m in second if "createSurface" in m)
            event = {"version": "v0.9.1", "action": {"name": "show_details", "surfaceId": surface,
                "sourceComponentId": "table", "timestamp": "2026-09-22T08:00:00Z",
                "context": {"period": "2026-08", "region": "华东"}}}
            last, messages = post("/api/action", event, token)
            self.assertEqual(last, token)
            data = next(m["updateDataModel"]["value"] for m in messages if "updateDataModel" in m)
            self.assertEqual(data["queries"]["q1"]["filters"], {"period": "2026-08", "region": "华东", "view": "details"})
            self.assertEqual(len(data["queries"]["q1"]["rows"]), 3)
            with self.assertRaises(HTTPError) as error:
                post("/api/query", {"query": "继续"}, "foreign-token")
            self.assertEqual(error.exception.code, 409)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

if __name__ == "__main__":
    unittest.main()
