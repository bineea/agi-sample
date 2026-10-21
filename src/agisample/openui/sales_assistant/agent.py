"""独立会话与流式生成轮次；只有完整生成的结果才提交历史。"""
from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

from .errors import AgentError

MAX_TURNS = 12
TURN_TIMEOUT = 180
SESSION_TTL = 1800
MAX_HISTORY_BYTES = 180_000

PACKAGE = Path(__file__).resolve().parent


@dataclass
class Session:
    history: list = field(default_factory=list)
    turns: int = 0
    touched: float = field(default_factory=time.monotonic)
    lock: object = field(default_factory=threading.Lock)


class AgentService:
    def __init__(self, model, today=None):
        self.model = model
        self.today = today or date.today()
        self.sessions = {}
        self._sessions_lock = threading.Lock()
        business = (PACKAGE / "agent_prompt.md").read_text(encoding="utf-8-sig")
        library_prompt_path = PACKAGE / "library_prompt.txt"
        if not library_prompt_path.is_file():
            raise ValueError("缺少 library_prompt.txt，请先在 web 目录执行 npm run generate-prompt。")
        library_prompt = library_prompt_path.read_text(encoding="utf-8-sig")
        self.system = (business.replace("__TODAY__", self.today.isoformat())
                       + "\n\n" + library_prompt)

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
        """校验并锁定会话，返回 (session_id, 事件生成器)。"""
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError("请输入一到两千字的问题。")
        session_id, session = self._session(session_id)
        return session_id, self._turn(session_id, session, text.strip())

    def _turn(self, session_id, session, text):
        """NDJSON 事件流：delta* → done | error。只在 done 时提交历史。"""
        try:
            if session.turns >= MAX_TURNS:
                raise AgentError("本会话已达到十二轮上限，请点击“新会话”。", 409)
            history = [*session.history, {"role": "user", "content": text}]
            deadline = time.monotonic() + TURN_TIMEOUT
            answer = ""
            for delta in self.model.complete_stream(
                    [{"role": "system", "content": self.system}, *history],
                    min(90, max(10, deadline - time.monotonic()))):
                if time.monotonic() > deadline:
                    raise AgentError("本轮处理超时，请缩小查询范围后重试。", 504)
                answer += delta
                yield {"type": "delta", "text": delta}
            if not answer.strip():
                raise AgentError("模型未返回有效内容。")
            answer_msg = {"role": "assistant", "content": answer}
            if len(json.dumps([*history, answer_msg], ensure_ascii=False)) > MAX_HISTORY_BYTES:
                raise AgentError("会话上下文已达上限，请点击“新会话”。", 409)
            session.history = [*history, answer_msg]
            session.turns += 1
            yield {"type": "done", "turns": session.turns}
        except AgentError as exc:
            yield {"type": "error", "error": str(exc), "status": exc.status}
        finally:
            session.touched = time.monotonic()
            session.lock.release()
            if not session.turns:
                with self._sessions_lock:
                    self.sessions.pop(session_id, None)
