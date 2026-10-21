<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";
import * as echarts from "echarts";

/** 通用 ECharts 容器：负责初始化、option 更新、尺寸自适应与销毁 */
const props = defineProps<{ option: echarts.EChartsOption }>();

const el = ref<HTMLDivElement>();
let chart: echarts.ECharts | null = null;
let observer: ResizeObserver | null = null;

onMounted(() => {
  if (!el.value) return;
  // 流式渲染时容器可能暂时为 0 宽，等拿到实际尺寸再初始化
  observer = new ResizeObserver(() => {
    if (!el.value) return;
    if (!chart && el.value.clientWidth > 0) {
      chart = echarts.init(el.value);
      chart.setOption(props.option);
    } else {
      chart?.resize();
    }
  });
  observer.observe(el.value);
});

watch(
  () => props.option,
  (option) => {
    chart?.setOption(option, { notMerge: true });
  },
  { deep: true },
);

onBeforeUnmount(() => {
  observer?.disconnect();
  chart?.dispose();
  chart = null;
});
</script>

<template>
  <div ref="el" class="echart"></div>
</template>

<style scoped>
.echart {
  width: 100%;
  height: 320px;
}
</style>
