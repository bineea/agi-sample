"""使用真实 SDK 的模拟 HTTP 传输验证示例，不访问远端接口。"""

import contextlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx2
from typesafe_sdk import RetryPolicy, TypeSafeClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agisample.integrations import typesafe_jev_demo as demo


RESPONSE = {
    "model": "jev-test",
    "usage": {"input_tokens": 100, "output_tokens": 20},
    "answers": {
        "department": {"type": "choice", "choice": "technical", "confidence": 0.9,
                       "probabilities": {"technical": 0.9, "billing": 0.1, "sales": 0.0}},
        "frustration": {"type": "score", "score": 1.2, "confidence": 0.8,
                        "legend": {"0": "平静", "1": "不满", "2": "愤怒"},
                        "probabilities": {"0": 0.0, "1": 0.8, "2": 0.2}},
        "is_urgent": {"type": "noul", "noul": 0.95},
    },
}


class TypeSafeDemoTests(unittest.TestCase):
    def run_cli(self, handler, argv=None):
        client = TypeSafeClient(
            api_key="test-key", base_url="https://api.typesafe.ai",
            transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0),
        )
        output, error = io.StringIO(), io.StringIO()
        with patch.object(demo, "load_dotenv"), patch.dict(demo.os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                patch.object(demo, "TypeSafeClient", return_value=client), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            code = demo.main(argv or [])
        return code, output.getvalue(), error.getvalue()

    def test_sdk_request_and_response(self):
        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/systemone")
            self.assertEqual(request.headers["Authorization"], "Bearer test-key")
            body = json.loads(request.content)
            self.assertEqual(body["state"], "接口失败")
            self.assertEqual(body["model"], "jev-latest")
            self.assertEqual({q["type"] for q in body["questions"].values()}, {"choice", "score", "noul"})
            return httpx2.Response(200, json=RESPONSE)
        code, output, error = self.run_cli(handle, ["--text", "接口失败"])
        self.assertEqual(code, 0, error)
        self.assertIn("technical", output)
        self.assertIn("1.2", output)
        self.assertIn("0.95", output)
        self.assertIn('"input_tokens": 100', output)

    def test_missing_key_and_empty_input_do_not_create_client(self):
        for args, key in [([], ""), (["--text", "  "], "test-key")]:
            with self.subTest(args=args), patch.object(demo, "load_dotenv"), \
                    patch.dict(demo.os.environ, {"TYPESAFE_API_KEY": key}), \
                    patch.object(demo, "TypeSafeClient") as client, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(demo.main(args), 2)
                client.assert_not_called()

    def test_api_error(self):
        code, output, error = self.run_cli(lambda request: httpx2.Response(401, json={"message": "invalid key"}))
        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        self.assertIn("HTTP 401", error)
        self.assertNotIn("test-key", error)

    def test_connection_error(self):
        def handle(request):
            raise httpx2.ConnectError("模拟网络故障", request=request)
        self.assertEqual(self.run_cli(handle)[0], 1)

    def test_invalid_response(self):
        self.assertEqual(self.run_cli(lambda request: httpx2.Response(200, json={"unexpected": True}))[0], 1)

    def test_missing_answer(self):
        body = dict(RESPONSE, answers={})
        code, _, error = self.run_cli(lambda request: httpx2.Response(200, json=body))
        self.assertEqual(code, 1)
        self.assertIn("缺少", error)


if __name__ == "__main__":
    unittest.main()
