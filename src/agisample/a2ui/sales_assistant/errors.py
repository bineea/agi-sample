"""对外可展示的中文 Agent 错误，禁止拼接模型供应商原始响应。"""

class AgentError(RuntimeError):
    def __init__(self, message, status=502):
        super().__init__(message)
        self.status = status
