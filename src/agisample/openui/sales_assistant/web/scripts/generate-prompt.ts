/**
 * 从组件库规格生成发给模型的 OpenUI 系统提示词，写入 ../library_prompt.txt。
 * 运行：npm run generate-prompt（需先 npm install）
 * 后端启动时读取该文件，与 agent_prompt.md 的业务规则拼接为完整系统提示词。
 */
import { createLibrary, defineComponent } from "@openuidev/lang-core";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildComponents } from "../src/openui/library/spec";

const { all } = buildComponents(defineComponent as any, () => null);
const library = createLibrary({ root: "Stack", components: all });

const prompt = library.prompt({
  tools: [
    {
      name: "query_sales",
      description:
        "只读查询虚构销售数据。返回标题、汇总指标（metrics: total/count/avg，金额均为已格式化字符串）、" +
        "区域汇总（regions[]）与订单明细（orders[]）。每次只查一个月份和一个区域，可在一个界面中写多个 Query 比较。",
      inputSchema: {
        type: "object",
        properties: {
          period: { type: "string", pattern: "^[0-9]{4}-(0[1-9]|1[0-2])$", description: "年月 YYYY-MM" },
          region: { type: "string", enum: ["全部", "华东", "华南", "华北", "西部"] },
          view: { type: "string", enum: ["summary", "details"], description: "summary 汇总 / details 明细" },
        },
        required: ["period", "region", "view"],
        additionalProperties: false,
      },
      outputSchema: {
        type: "object",
        properties: {
          title: { type: "string" },
          filters: { type: "object" },
          metrics: {
            type: "object",
            properties: {
              total: { type: "string", description: "销售额，已格式化如 ¥12,880" },
              count: { type: "number", description: "订单数" },
              avg: { type: "string", description: "平均订单金额，已格式化" },
            },
          },
          regions: {
            type: "array",
            items: {
              type: "object",
              properties: {
                region: { type: "string" },
                order_count: { type: "number" },
                total: { type: "string" },
                total_cents: { type: "number" },
                total_value: { type: "number", description: "销售额（元），图表 values 用它" },
              },
            },
          },
          orders: {
            type: "array",
            items: {
              type: "object",
              properties: {
                id: { type: "string" },
                ordered_on: { type: "string" },
                region: { type: "string" },
                customer: { type: "string" },
                product: { type: "string" },
                amount: { type: "string" },
              },
            },
          },
        },
      },
      annotations: { readOnlyHint: true },
    },
  ],
  toolExamples: [
    `sales = Query("query_sales", {period: "2026-10", region: "全部", view: "summary"}, {title: "加载中", metrics: {total: "…", count: 0, avg: "…"}, regions: []})\nroot = Stack([head, stats, chart], "column", "l")\nhead = TextContent(sales.title)\nstats = Stack([k1, k2], "row", "m")\nk1 = StatCard("销售额", sales.metrics.total)\nk2 = StatCard("订单数", "" + sales.metrics.count)\nchart = BarChart("区域销售额（元）", sales.regions.region, sales.regions.total_value)`,
    `$region = "全部"\nsales = Query("query_sales", {period: "2026-10", region: $region, view: "summary"}, {title: "加载中", metrics: {total: "…", count: 0, avg: "…"}, regions: []})\nroot = Stack([head, picker, stats], "column", "m")\nhead = TextContent(sales.title)\npicker = FormControl("区域（切换后自动重新查询）", Select("region", [SelectItem("all", "全部"), SelectItem("east", "华东")], null, $region))\nstats = StatCard("销售额", sales.metrics.total)`,
    `details = Query("query_sales", {period: "2026-10", region: "华东", view: "details"}, {title: "加载中", orders: []})\nroot = Stack([tbl], "column", "m")\ntbl = DataTable(details.title, [{key: "id", label: "订单号"}, {key: "customer", label: "客户"}, {key: "amount", label: "金额", align: "right"}], details.orders, "暂无订单")`,
  ],
});

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "library_prompt.txt");
writeFileSync(out, prompt, "utf-8");
console.log("已生成", out, `(${prompt.length} 字符)`);
