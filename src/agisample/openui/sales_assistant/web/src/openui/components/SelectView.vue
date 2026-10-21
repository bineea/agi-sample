<script setup lang="ts">
import { computed } from "vue";
import { useIsStreaming, type ComponentRenderProps } from "@openuidev/vue-lang";
import { useStateField } from "../useStateField";

const p = defineProps<
  ComponentRenderProps<{
    name: string;
    items: Array<{ props: { value: string; label?: string } }>;
    placeholder?: string;
    value?: unknown;
  }>
>();

const isStreaming = useIsStreaming();
const field = useStateField(
  () => p.props.name,
  () => p.props.value,
  "Select",
);

// items 是解析后的子组件元素（SubComponentOf），取其 props
const options = computed(() =>
  (p.props.items ?? [])
    .filter((it) => it?.props?.value)
    .map((it) => ({ value: it.props.value, label: it.props.label || it.props.value })),
);
</script>

<template>
  <el-select
    :model-value="(field.value.value as string) ?? ''"
    :placeholder="p.props.placeholder ?? '请选择'"
    :disabled="isStreaming"
    style="width: 100%"
    @change="(v: string) => field.setValue(v, true)"
  >
    <el-option v-for="opt in options" :key="opt.value" :value="opt.value" :label="opt.label" />
  </el-select>
</template>
