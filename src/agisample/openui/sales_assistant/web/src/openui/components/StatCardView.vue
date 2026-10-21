<script setup lang="ts">
import type { ComponentRenderProps } from "@openuidev/vue-lang";

defineProps<ComponentRenderProps<{ label: string; value: string; hint?: string }>>();
</script>

<template>
  <el-card shadow="never" class="stat-card">
    <div class="label">{{ props.label }}</div>
    <div class="value">{{ props.value }}</div>
    <div v-if="props.hint" class="hint">{{ props.hint }}</div>
  </el-card>
</template>

<style scoped>
.stat-card {
  flex: 1;
}
.label {
  font-size: 13px;
  color: #86909c;
}
.value {
  margin-top: 4px;
  font-size: 26px;
  font-weight: 700;
}
.hint {
  margin-top: 4px;
  font-size: 12px;
  color: #00b42a;
}
</style>
