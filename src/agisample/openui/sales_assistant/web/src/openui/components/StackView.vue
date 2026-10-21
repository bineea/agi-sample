<script setup lang="ts">
import type { ComponentRenderProps } from "@openuidev/vue-lang";
import { NodeSlot } from "../NodeSlot";

defineProps<ComponentRenderProps<{ children: unknown; direction?: string; gap?: string }>>();

const gapMap: Record<string, string> = { s: "8px", m: "16px", l: "24px" };
</script>

<template>
  <div
    class="openui-stack"
    :style="{
      flexDirection: props.direction === 'row' ? 'row' : 'column',
      gap: gapMap[props.gap ?? 'm'] ?? '16px',
    }"
  >
    <NodeSlot :value="props.children" :render-node="renderNode" />
  </div>
</template>

<style scoped>
.openui-stack {
  display: flex;
  width: 100%;
}
.openui-stack > * {
  min-width: 0;
  flex: 1 1 0%;
}
</style>
