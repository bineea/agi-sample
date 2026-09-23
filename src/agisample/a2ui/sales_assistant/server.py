"""本地 HTTP 服务：默认真实 Agent，可显式选择规则 demo。"""
import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from urllib.parse import urlsplit

from .errors import AgentError
from .service import SalesService

WEB = Path(__file__).with_name("web")
STATIC = {"/": ("index.html", "text/html"), "/styles.css": ("styles.css", "text/css"),
          **{"/" + name: (name, "text/javascript") for name in ("app.mjs", "state.mjs", "renderer.mjs", "session.mjs")},
          "/catalog.json": ("catalog.json", "application/json")}
MAX_BODY = 16_384


def create_server(host="127.0.0.1", port=8765, *, today: date | None = None, mode="agent", service=None):
    if mode not in ("agent", "demo"):
        raise ValueError("模式仅支持 agent 或 demo。")
    if service is None:
        if mode == "agent":
            from .agent import AgentService
            from .llm import ChatModel, ModelConfig
            service = AgentService(ChatModel(ModelConfig.from_env()), today)
        else:
            service = SalesService(today)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send_body(self, status, body, content_type, session_id=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if session_id:
                self.send_header("X-A2UI-Session", session_id)
            self.end_headers()
            self.wfile.write(body)

        def error(self, status, message):
            self.send_body(status, json.dumps({"error": message}, ensure_ascii=False).encode(), "application/json")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/info":
                info = {"mode": mode, "model": service.model.model if mode == "agent" else None,
                        "sample": True, "referenceDate": service.today.isoformat()}
                self.send_body(200, json.dumps(info, ensure_ascii=False).encode(), "application/json")
                return
            resource = STATIC.get(path)
            if not resource:
                self.error(404, "资源不存在。")
                return
            name, content_type = resource
            self.send_body(200, (WEB / name).read_bytes(), content_type)

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/query", "/api/action"):
                self.error(404, "接口不存在。")
                return
            if self.headers.get_content_type() != "application/json":
                self.error(415, "请求必须使用 application/json。")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.error(400, "无效的请求长度。")
                return
            if not 0 < size <= MAX_BODY:
                self.error(413 if size > MAX_BODY else 400, "请求为空或超过 16 KB。")
                return
            try:
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("请求内容必须是 JSON 对象。")
                session_id = self.headers.get("X-A2UI-Session")
                if mode == "agent":
                    result = (service.query(payload.get("query"), session_id) if path == "/api/query"
                              else service.action(payload, session_id))
                    messages, session_id = result.messages, result.session_id
                else:
                    messages = service.query(payload.get("query")) if path == "/api/query" else service.action(payload)
                    session_id = None
            except (json.JSONDecodeError, UnicodeError):
                self.error(400, "请求必须是有效的 UTF-8 JSON 内容。")
                return
            except AgentError as exc:
                self.error(exc.status, str(exc))
                return
            except ValueError as exc:
                self.error(400, str(exc))
                return
            except (TimeoutError, socket.timeout):
                self.error(408, "读取请求超时，请重试。")
                return
            body = "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in messages).encode()
            self.send_body(200, body, "application/x-ndjson", session_id)

        def log_message(self, fmt, *args):
            print(f"[HTTP] {self.address_string()} {fmt % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    server.agent_service = service
    return server


def main():
    parser = argparse.ArgumentParser(description="启动 A2UI 销售查询助手：大模型工具调用与动态界面")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8765, help="端口，默认 8765")
    parser.add_argument("--mode", choices=("agent", "demo"), default="agent", help="默认 agent；demo 是无模型规则演示")
    args = parser.parse_args()
    try:
        server = create_server(args.host, args.port, mode=args.mode)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    print(f"A2UI 销售查询助手：http://{args.host}:{server.server_port}，模式：{args.mode}（Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()
        if args.mode == "agent":
            server.agent_service.model.close()


if __name__ == "__main__":
    main()
