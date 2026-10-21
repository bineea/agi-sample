<script setup lang="ts">
import { computed } from "vue";
import type { ComponentRenderProps } from "@openuidev/vue-lang";
import type { EChartsOption } from "echarts";
import EChart from "../../components/EChart.vue";

const p = defineProps<
  ComponentRenderProps<{ title: string; items: Array<{ name: string; value: number }> }>
>();

const option = computed<EChartsOption>(() => ({
  title: { text: p.props.title, textStyle: { fontSize: 15 } },
  tooltip: { trigger: "item" },
  legend: { bottom: 0 },
  series: [
    {
      type: "pie",
      radius: ["38%", "62%"],
      center: ["50%", "52%"],
      data: (p.props.items ?? []).map((it) => ({ name: it.name, value: it.value })),
    },
  ],
}));
</script>

<template>
  <el-card shadow="never">
    <EChart :option="option" />
  </el-card>
</template>
