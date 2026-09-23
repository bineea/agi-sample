# A2UI 销售查询示例设计

用户已确认设计并授权实现。前端负责 A2UI 渲染、数据绑定和事件回传；后端负责规则意图解析、SQLite 只读查询和 A2UI 消息生成。

Python 3.11+ 标准库 HTTP + SQLite 内存库，浏览器原生 ES modules，无需新依赖或密钥。支持本月、上月、YYYY-MM，四区域汇总和订单明细。数据为虚构，默认规则模式，不使用 LLM 或任意 SQL。

协议使用 A2UI v0.9.1，createSurface/updateDataModel/updateComponents/deleteSurface 与 action；POST 返回 NDJSON。自定义目录 urn:agisample:a2ui:sales:1 定义 Text/Row/Column/Card/Button/DataTable。提供有限教学渲染器，非完整官方 Basic Catalog。表格列与行通过后端描述和数据绑定生成。

页面为左侧查询、右侧结果的分析工作台，窄屏单列。白色、浅蓝灰、深蓝、蓝色、青色；Segoe UI/Microsoft YaHei。原始消息折叠展示，虚构数据和支持的查询范围明确可见。

验收覆盖汇总、跨年月、区域筛选、空数据、无效输入、独立客户端、数据模型更新、流拆包、HTML 安全文本化、真实 HTTP 和浏览器交互。
