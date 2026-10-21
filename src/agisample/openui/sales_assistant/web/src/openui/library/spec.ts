import { markReactive, tagSchemaId } from "@openuidev/lang-core";
import { z } from "zod/v4";

/**
 * 组件库规格（单一事实来源）：只含 zod schema 与描述，不含渲染实现。
 * - 前端运行时：library/index.ts 用 @openuidev/vue-lang 的 defineComponent + Vue 视图构建；
 * - 提示词生成：scripts/generate-prompt.ts 用 @openuidev/lang-core + 空实现构建，
 *   生成发给模型的系统提示词（library_prompt.txt）。
 * 两处共用本文件，保证"模型看到的组件词汇表"与"前端能渲染的组件"永远一致。
 */

type DefineFn = (config: { name: string; description: string; props: object; component: unknown }) => any;

/** 标记 schema 字段可接收 $variable 绑定（等价于 react-lang 的 reactive()） */
function reactive<T extends object>(schema: T): T {
  markReactive(schema);
  return schema;
}

const actionPropSchema = z.any();
tagSchemaId(actionPropSchema, "ActionExpression");

export function buildComponents(defineComponent: DefineFn, componentFor: (name: string) => unknown) {
  const def = (name: string, description: string, props: object) =>
    defineComponent({ name, description, props, component: componentFor(name) });

  // ── 布局 ──────────────────────────────────────────────
  const Stack = def("Stack", "布局容器，按方向排列子组件", z.object({
    children: z.array(z.any()),
    direction: z.enum(["column", "row"]).optional(),
    gap: z.enum(["s", "m", "l"]).optional(),
  }));

  // ── 基础展示 ──────────────────────────────────────────
  const TextContent = def("TextContent", "文本段落", z.object({ text: z.string() }));

  const StatCard = def("StatCard", "指标卡片：标签 + 数值 + 辅助说明", z.object({
    label: z.string(),
    value: z.string(),
    hint: z.string().optional(),
  }));

  // ── 图表 ──────────────────────────────────────────────
  const BarChart = def("BarChart", "柱状图", z.object({
    title: z.string(),
    labels: z.array(z.string()),
    values: z.array(z.number()),
  }));

  const PieChart = def("PieChart", "饼图", z.object({
    title: z.string(),
    items: z.array(z.object({ name: z.string(), value: z.number() })),
  }));

  const DataTable = def("DataTable", "数据表格：行绑定查询结果数组，列用字面量定义", z.object({
    caption: z.string().optional(),
    columns: z.array(z.object({
      key: z.string(),
      label: z.string(),
      align: z.enum(["left", "center", "right"]).optional(),
    })),
    rows: z.array(z.record(z.string(), z.any())),
    emptyText: z.string().optional(),
  }));

  // ── 表单字段 ──────────────────────────────────────────
  const Input = def("Input", "单行文本输入框", z.object({
    name: z.string(),
    placeholder: z.string().optional(),
    type: z.enum(["text", "number", "password"]).optional(),
    value: reactive(z.string().optional()),
  }));

  const SelectItem = def("SelectItem", "下拉选项", z.object({
    value: z.string(),
    label: z.string().optional(),
  }));

  const Select = def("Select", "下拉选择框", z.object({
    name: z.string(),
    items: z.array(SelectItem.ref),
    placeholder: z.string().optional(),
    value: reactive(z.string().optional()),
  }));

  const FormControl = def("FormControl", "表单项：标签 + 一个输入控件", z.object({
    label: z.string(),
    input: z.union([Input.ref, Select.ref]),
    hint: z.string().optional(),
  }));

  // ── 按钮与动作 ────────────────────────────────────────
  const Button = def("Button", "按钮，点击执行 action（Action([...]) 表达式）", z.object({
    label: z.string(),
    action: actionPropSchema.optional(),
    variant: z.enum(["primary", "secondary"]).optional(),
  }));

  const Buttons = def("Buttons", "按钮组", z.object({
    buttons: z.array(Button.ref),
  }));

  const Form = def("Form", "表单容器：字段列表 + 按钮组", z.object({
    name: z.string(),
    fields: z.array(FormControl.ref).default([]),
    buttons: Buttons.ref,
  }));

  const all = [
    Stack, TextContent, StatCard, BarChart, PieChart, DataTable,
    Input, SelectItem, Select, FormControl, Button, Buttons, Form,
  ];

  return { Stack, TextContent, StatCard, BarChart, PieChart, DataTable, Input, SelectItem, Select, FormControl, Button, Buttons, Form, all };
}
