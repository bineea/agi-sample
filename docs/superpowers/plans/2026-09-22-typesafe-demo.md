# TypeSafe 使用示例实施计划

目标：落实用户已确认的 Python SDK 客服工单分析示例。

设计：在 `src/agisample/integrations/typesafe_demo.py` 提供独立命令行入口，
使用 Choice、Score、Noul 同时分析一条工单；从环境或项目根目录 `.env`
加载密钥，环境变量优先。默认模型为 `jev-latest`，支持文本和模型参数。
使用上下文管理器关闭客户端，错误写入标准错误并返回非零退出码。

约束：使用现有 Python 3.11 虚拟环境；中文说明；不修改已有无关文件。

- [x] 实现 SDK 示例及命令行参数，固定 `typesafe-sdk==0.7.1` 依赖。
- [x] 补充 README 的 PowerShell 运行命令及三种输出的解释。
- [x] 使用真实 SDK 和 MockTransport 验证请求、响应及错误路径，无需外部服务。
- [x] 检查帮助命令、差异和验证结果；单独说明真实 API 验收状态。

验证结果：6 项离线测试通过；帮助命令成功；`git diff --check` 通过。
真实 API 未调用，远端效果未验收。
