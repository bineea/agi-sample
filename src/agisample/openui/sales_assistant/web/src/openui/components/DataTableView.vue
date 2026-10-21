<script setup lang="ts">
import type { ComponentRenderProps } from "@openuidev/vue-lang";

defineProps<
  ComponentRenderProps<{
    caption?: string;
    columns: Array<{ key: string; label: string; align?: string }>;
    rows: Array<Record<string, unknown>>;
    emptyText?: string;
  }>
>();
</script>

<template>
  <el-card shadow="never">
    <div v-if="props.caption" class="caption">{{ props.caption }}</div>
    <el-table :data="props.rows ?? []" stripe size="small">
      <el-table-column
        v-for="col in props.columns ?? []"
        :key="col.key"
        :prop="col.key"
        :label="col.label"
        :align="(col.align as any) ?? 'left'"
      />
      <template #empty>{{ props.emptyText ?? "暂无数据" }}</template>
    </el-table>
  </el-card>
</template>

<style scoped>
.caption {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 10px;
}
</style>
