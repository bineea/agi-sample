<script setup lang="ts">
import type { ComponentRenderProps } from "@openuidev/vue-lang";
import { NodeSlot } from "../NodeSlot";

defineProps<ComponentRenderProps<{ label: string; input: unknown; hint?: string }>>();
</script>

<template>
  <el-form-item :label="props.label" label-position="top" class="openui-form-control">
    <NodeSlot :value="props.input" :render-node="renderNode" />
    <div v-if="props.hint" class="hint">{{ props.hint }}</div>
  </el-form-item>
</template>

<style scoped>
.openui-form-control {
  margin-bottom: 0;
  width: 100%;
}
.hint {
  font-size: 12px;
  color: #86909c;
  margin-top: 4px;
}
</style>
