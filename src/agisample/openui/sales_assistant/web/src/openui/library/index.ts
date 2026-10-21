import { createLibrary, defineComponent } from "@openuidev/vue-lang";

import StackView from "../components/StackView.vue";
import TextContentView from "../components/TextContentView.vue";
import StatCardView from "../components/StatCardView.vue";
import BarChartView from "../components/BarChartView.vue";
import PieChartView from "../components/PieChartView.vue";
import DataTableView from "../components/DataTableView.vue";
import FormView from "../components/FormView.vue";
import FormControlView from "../components/FormControlView.vue";
import InputView from "../components/InputView.vue";
import SelectView from "../components/SelectView.vue";
import ButtonView from "../components/ButtonView.vue";
import ButtonsView from "../components/ButtonsView.vue";
import { buildComponents } from "./spec";

/** 组件名 → Vue 渲染实现；schema 统一来自 spec.ts */
const views: Record<string, unknown> = {
  Stack: StackView,
  TextContent: TextContentView,
  StatCard: StatCardView,
  BarChart: BarChartView,
  PieChart: PieChartView,
  DataTable: DataTableView,
  Input: InputView,
  Select: SelectView,
  SelectItem: { setup: () => () => null }, // SelectItem 仅承载数据，不渲染
  FormControl: FormControlView,
  Button: ButtonView,
  Buttons: ButtonsView,
  Form: FormView,
};

export const {
  Stack, TextContent, StatCard, BarChart, PieChart, DataTable,
  Input, SelectItem, Select, FormControl, Button, Buttons, Form, all,
} = buildComponents(defineComponent as any, (name) => views[name]);

export const demoLibrary = createLibrary({
  root: "Stack",
  components: all,
});
