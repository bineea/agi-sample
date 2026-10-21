import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5174,
    proxy: {
      // 开发模式下把 API 请求转发给 Python 后端（python -m agisample.openui.sales_assistant）
      "/api": {
        target: "http://127.0.0.1:8766",
        changeOrigin: true,
      },
    },
  },
});
