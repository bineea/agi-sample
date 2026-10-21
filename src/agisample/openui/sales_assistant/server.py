"""本地 HTTP 服务：流式对话接口 + 只读业务工具 + 静态前端。"""
import argparse
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
from urllib.parse import urlsplit

from .agent_tools import run_query
from .errors import AgentError

PACKAGE = Path(__file__).resolve().parent
DIST = PACKAGE / "web" / "dist"
MAX_BODY = 16_384
SESSION_HEADER = "X-OpenUI-Session"

CONTENT_TYPES = {
    ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
    ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png",
    ".woff2": "font/woff2", ".ico": "image/x-icon",
}


def create_server(host="127.0.0.1", port=8766, *, today: date | None = None, mode="agent", service=None):
    if mode not in ("agent", "demo"):
        raise ValueError("模式仅支持 agent 或 demo。")
    if service is None:
        from .agent import AgentService
        if mode == "agent":
            from .llm import ChatModel, ModelConfig
            service = AgentService(ChatModel(ModelConfig.from_env()), today)
        else:
            from .demo import DemoModel
            service = AgentService(DemoModel(), today)
    reference_today = today or date.today()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def _headers(self, status, content_type, session_id=None, length=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if length is not None:
                self.send_header("Content-Length", str(length))
            if session_id:
                self.send_header(SESSION_HEADER, session_id)
            self.end_headers()

        def send_json(self, status, payload, session_id=None):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self._headers(status, "application/json", session_id, len(body))
            self.wfile.write(body)

        def error(self, status, message):
            self.send_json(status, {"error": message})

        def read_json(self):
            if self.headers.get_content_type() != "application/json":
                raise AgentError("请求必须使用 application/json。", 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise AgentError("无效的请求长度。", 400) from None
            if not 0 < size <= MAX_BODY:
                raise AgentError("请求为空或超过 16 KB。", 413 if size > MAX_BODY else 400)
            try:
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeError):
                raise AgentError("请求必须是有效的 UTF-8 JSON 内容。", 400) from None
            if not isinstance(payload, dict):
                raise AgentError("请求内容必须是 JSON 对象。", 400)
            return payload

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/info":
                self.send_json(200, {"mode": mode, "model": service.model.model,
                                     "sample": True, "referenceDate": service.today.isoformat()})
                return
            # 静态资源：优先 web/dist（npm run build 产物），否则提示开发模式
            if not DIST.is_dir():
                self.error(404, "前端未构建。开发模式请在 web 目录运行 npm run dev。")
                return
            target = DIST / ("index.html" if path == "/" else path.lstrip("/"))
            try:
                resolved = target.resolve()
                if not resolved.is_file() or DIST.resolve() not in resolved.parents:
                    raise FileNotFoundError
            except (OSError, ValueError):
                self.error(404, "资源不存在。")
                return
            body = resolved.read_bytes()
            content_type = CONTENT_TYPES.get(resolved.suffix, "application/octet-stream")
            self._headers(200, content_type, length=len(body))
            self.wfile.write(body)

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                payload = self.read_json()
                if path == "/api/tools/query_sales":
                    self.send_json(200, run_query(payload.get("arguments"), reference_today))
                    return
                if path == "/api/chat":
                    self.chat(payload)
                    return
                self.error(404, "接口不存在。")
            except AgentError as exc:
                self.error(exc.status, str(exc))
            except ValueError as exc:
                self.error(400, str(exc))
            except (TimeoutError, socket.timeout):
                self.error(408, "读取请求超时，请重试。")

        def chat(self, payload):
            # 会话与输入校验在响应开始之前完成，之后切换为 NDJSON 流
            session_id, events = service.query(payload.get("query"),
                                               self.headers.get(SESSION_HEADER))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(SESSION_HEADER, session_id)
            self.end_headers()
            try:
                for event in events:
                    self.wfile.write(json.dumps(event, ensure_ascii=False).encode() + b"\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # 客户端中断：会话锁已在生成器 finally 中释放

        def log_message(self, fmt, *args):
            print(f"[HTTP] {self.address_string()} {fmt % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    server.agent_service = service
    return server


def main():
    parser = argparse.ArgumentParser(description="启动 OpenUI 销售查询助手：大模型生成动态界面（Vue 渲染）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8766, help="端口，默认 8766")
    parser.add_argument("--mode", choices=("agent", "demo"), default="agent",
                        help="默认 agent；demo 是离线预置响应，仅用于无密钥学习")
    args = parser.parse_args()
    try:
        server = create_server(args.host, args.port, mode=args.mode)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    print(f"OpenUI 销售查询助手：http://{args.host}:{server.server_port}，模式：{args.mode}（Ctrl+C 停止）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()
        server.agent_service.model.close()


if __name__ == "__main__":
    main()
