"""共享的 ChatOpenAI 模型工厂。

模型名、密钥、接口地址全部通过环境变量配置（与项目其他示例保持一致）：
- API_KEY: 必填，OpenAI 兼容接口的密钥
- MODEL: 模型名称
- BASE_URL: 可选，自定义接口地址
"""

import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI


def build_chat_model() -> ChatOpenAI:
    """按环境变量构建 ChatOpenAI；缺少 API_KEY 时直接报错。"""
    load_dotenv(find_dotenv())
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 API_KEY")
    return ChatOpenAI(
        model=os.getenv("MODEL", ""),
        api_key=api_key,
        base_url=os.getenv("BASE_URL") or None,
        temperature=0,
    )
