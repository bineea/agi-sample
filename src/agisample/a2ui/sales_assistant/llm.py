"""OpenAI 兼容 Chat Completions 接口；配置只在服务端读取。"""
from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError, APITimeoutError

from .errors import AgentError


@dataclass(frozen=True)
class ModelConfig:
    api_key: str = field(repr=False)
    model: str
    base_url: str = "https://api.openai.com/v1"

    @classmethod
    def from_env(cls, env_file=None):
        path = Path(env_file) if env_file is not None else Path(__file__).resolve().parents[4] / ".env"
        values = dotenv_values(path) if path.is_file() else {}
        def select(*names, default=""):
            # 显式环境变量优先于文件，即使使用的是兼容别名。
            for source in (os.environ, values):
                for name in names:
                    if (source.get(name) or "").strip():
                        return source[name].strip()
            return default
        key = select("OPENAI_API_KEY", "API_KEY")
        model = select("OPENAI_MODEL", "MODEL")
        base = select("OPENAI_BASE_URL", "BASE_URL", default="https://api.openai.com/v1").rstrip("/")
        if not key or not model:
            raise ValueError("Agent 模式需要 OPENAI_API_KEY 和 OPENAI_MODEL；兼容服务请同时设置 OPENAI_BASE_URL（包含 /v1）。")
        parsed = urlsplit(base)
        if parsed.scheme not in ("https", "http") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OPENAI_BASE_URL 必须是合法 HTTP(S) 接口地址，不得包含凭据或查询参数。")
        return cls(key, model, base)


class ChatModel:
    def __init__(self, config: ModelConfig, *, client=None):
        self.model = config.model
        self.client = client or OpenAI(api_key=config.api_key, base_url=config.base_url,
                                       timeout=45, max_retries=0)

    def complete(self, messages, tools, timeout):
        try:
            result = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto",
                response_format={"type": "json_object"}, max_completion_tokens=6000,
                timeout=timeout)
        except APITimeoutError:
            raise AgentError("模型接口超时，请稍后重试。", 504) from None
        except OpenAIError:
            raise AgentError("模型接口调用失败，请检查接口地址、模型权限、额度及 tool calling/JSON 输出支持。") from None
        if not result.choices:
            raise AgentError("模型未返回有效响应。")
        choice = result.choices[0]
        if choice.finish_reason in ("length", "content_filter") or choice.message.refusal:
            raise AgentError("模型输出被截断或拒绝，请缩小查询范围后重试。")
        message = choice.message
        output = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            output["tool_calls"] = [call.model_dump(exclude_none=True) for call in message.tool_calls]
        return output

    def close(self):
        self.client.close()
