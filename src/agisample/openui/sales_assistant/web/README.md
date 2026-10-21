# OpenUI 销售助手前端

Vue 3 + `@openuidev/vue-lang` 前端，是 `src/agisample/openui/sales_assistant` 的 web 子工程。
使用方式与架构说明见上级目录的 [README.md](../README.md)。

```bash
npm install
npm run dev              # 开发（代理 /api → 127.0.0.1:8766）
npm run generate-prompt  # 组件 schema 变更后重新生成 ../library_prompt.txt
npm run build            # 产物到 dist/，由 Python 后端托管
```
