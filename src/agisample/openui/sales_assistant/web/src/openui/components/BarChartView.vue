<script setup lang="ts">
import { computed } from "vue";
import type { ComponentRenderProps } from "@openuidev/vue-lang";
import type { EChartsOption } from "echarts";
import EChart from "../../components/EChart.vue";

const p = defineProps<
  ComponentRenderProps<{ title: string; labels: string[]; values: number[] }>
>();

const option = computed<EChartsOption>(() => ({
  title: { text: p.props.title, textStyle: { fontSize: 15 } },
  tooltip: {},
  grid: { left: 48, right: 16, top: 48, bottom: 32 },
  xAxis: { type: "category", data: p.props.labels ?? [] },
  yAxis: { type: "value" },
  series: [{ type: "bar", data: p.props.values ?? [], itemStyle: { color: "#165dff", borderRadius: [4, 4, 0, 0] } }],
}));
</script>

<template>
  <el-card shadow="never">
    <EChart :option="option" />
  </el-card>
</template>
