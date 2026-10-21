<script setup lang="ts">
import { useIsStreaming, type ComponentRenderProps } from "@openuidev/vue-lang";
import { useStateField } from "../useStateField";

const p = defineProps<
  ComponentRenderProps<{ name: string; placeholder?: string; type?: string; value?: unknown }>
>();

const isStreaming = useIsStreaming();
const field = useStateField(
  () => p.props.name,
  () => p.props.value,
  "Input",
);
</script>

<template>
  <el-input
    :model-value="(field.value.value as string) ?? ''"
    :placeholder="p.props.placeholder"
    :type="p.props.type === 'number' ? 'number' : 'text'"
    :disabled="isStreaming"
    @input="(v: string) => field.setValue(v, false)"
    @blur="() => field.setValue(field.value.value, true)"
  />
</template>
