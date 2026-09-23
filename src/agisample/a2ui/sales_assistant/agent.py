"""有界工具循环与独立会话；只有成功校验的整轮结果才提交。"""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
import threading
import time
from uuid import uuid4
from .agent_tools import QUERY_TOOL, run_query
from .agent_ui import UIValidator
from .errors import AgentError

MAX_TURNS = 12
MAX_TOOL_CALLS = 4
MAX_MODEL_CALLS = 6
TURN_TIMEOUT = 180
SESSION_TTL = 1800


@dataclass
class Session:
    surface_id: str = field(default_factory=lambda: "sales-" + uuid4().hex)
    history: list = field(default_factory=list)
    components: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    turns: int = 0
    touched: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.Lock)


@dataclass
class TurnResult:
    messages: list
    session_id: str


class AgentService:
    mode = "agent"

    def __init__(self, model, today=None):
        self.model = model
        self.today = today or date.today()
        self.sessions = {}
        self._sessions_lock = threading.Lock()
        self.validator = UIValidator()
        prompt = Path(__file__).with_name("agent_prompt.md").read_text(encoding="utf-8-sig")
        self.system = prompt.replace("__TODAY__", self.today.isoformat())

    def _session(self, session_id):
        with self._sessions_lock:
            now = time.monotonic()
            for key, session in list(self.sessions.items()):
                if not session.lock.locked() and now - session.touched > SESSION_TTL:
                    del self.sessions[key]
            if session_id:
                if not isinstance(session_id, str) or session_id not in self.sessions:
                    raise AgentError("会话不存在或已过期，请点击“新会话”。", 409)
                session = self.sessions[session_id]
            else:
                if len(self.sessions) >= 64:
                    raise AgentError("演示会话数量已达上限，请稍后重试或重启服务。", 429)
                session_id = uuid4().hex
                session = Session()
                self.sessions[session_id] = session
            if not session.lock.acquire(blocking=False):
                raise AgentError("当前会话正在处理请求，请等待完成。", 409)
            session.touched = now
            return session_id, session

    def query(self, text, session_id=None):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError("请输入一到两千字的问题。")
        return self._turn(text.strip(), session_id)

    def action(self, message, session_id=None):
        if not session_id:
            raise ValueError("点击操作需要已有会话。")
        return self._turn(None, session_id, action=message)

    def _turn(self, text, session_id, action=None):
        session_id, session = self._session(session_id)
        try:
            if session.turns >= MAX_TURNS:
                raise AgentError("本会话已达到十二轮上限，请点击“新会话”。", 409)
            if action is not None:
                event = self.validator.validate_action(action, session.surface_id, session.components, session.data)
                text = "用户点击当前界面，请按事件查询并重新生成界面：" + json.dumps(event, ensure_ascii=False)
            history = deepcopy(session.history)
            history.append({"role": "user", "content": text})
            queries, steps = {}, []
            attempts, repaired = 0, False
            deadline = time.monotonic() + TURN_TIMEOUT
            for _ in range(MAX_MODEL_CALLS):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AgentError("本轮处理超时，请缩小查询范围后重试。", 504)
                answer = self.model.complete([{"role": "system", "content": self.system}, *history],
                                             [QUERY_TOOL], min(45, remaining))
                if not isinstance(answer, dict) or answer.get("role") != "assistant":
                    raise AgentError("模型响应格式不正确。")
                calls = answer.get("tool_calls") or []
                history.append(answer)
                if calls:
                    if not isinstance(calls, list) or attempts + len(calls) > MAX_TOOL_CALLS:
                        raise AgentError("本轮工具调用次数超过上限，请缩小查询范围。")
                    for call in calls:
                        attempts += 1
                        if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                            raise AgentError("模型工具调用缺少有效标识。")
                        function = call.get("function", {})
                        try:
                            if not isinstance(function, dict) or function.get("name") != "query_sales":
                                raise ValueError("只允许调用 query_sales，不提供 SQL 或其他工具。")
                            arguments = json.loads(function.get("arguments", ""))
                            data = run_query(arguments, self.today)
                            query_id = "q" + str(len(queries) + 1)
                            queries[query_id] = data
                            output = {"queryId": query_id, "data": data, "bindingPrefix": "/queries/" + query_id}
                            steps.append({"tool": "query_sales", "arguments": arguments, "status": "成功",
                                          "queryId": query_id, "orderCount": data["metrics"]["count"]})
                        except (ValueError, TypeError):
                            output = {"error": "工具参数不合法；只允许 query_sales(period:YYYY-MM, region:全部/华东/华南/华北/西部, view:summary/details)。"}
                            steps.append({"tool": "query_sales", "status": "拒绝", "error": output["error"]})
                        history.append({"role": "tool", "tool_call_id": call["id"],
                                        "content": json.dumps(output, ensure_ascii=False)})
                    continue
                model_data = {"queries": queries}
                try:
                    reply, components = self.validator.parse(answer.get("content"), model_data, session.surface_id)
                except ValueError as exc:
                    if repaired:
                        raise AgentError("模型生成的界面连续两次未通过校验，请重试或更换模型。") from None
                    repaired = True
                    history.append({"role": "user", "content": "界面校验失败：" + str(exc) + " 请仅修正 JSON 布局，保留已取得的数据绑定。"})
                    continue
                model_data["agent"] = {"reply": reply, "steps": steps, "mode": "agent", "model": self.model.model}
                messages = self.validator.messages(session.surface_id, model_data, components, initial=session.turns == 0)
                if len(json.dumps(history, ensure_ascii=False)) > 180_000:
                    raise AgentError("会话上下文已达上限，请点击“新会话”。", 409)
                session.history, session.data, session.components = history, model_data, components
                session.turns += 1
                return TurnResult(messages, session_id)
            raise AgentError("模型未在限定轮数内完成，请简化问题后重试。")
        finally:
            session.touched = time.monotonic()
            session.lock.release()
            if not session.turns:
                with self._sessions_lock:
                    self.sessions.pop(session_id, None)
