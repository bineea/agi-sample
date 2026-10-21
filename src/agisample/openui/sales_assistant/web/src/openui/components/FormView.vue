<script setup lang="ts">
import { computed } from "vue";
import { provideFormName, type ComponentRenderProps } from "@openuidev/vue-lang";
import { NodeSlot } from "../NodeSlot";

const p = defineProps<ComponentRenderProps<{ name: string; fields: unknown; buttons: unknown }>>();

// 向子级字段组件提供当前表单名，字段读写都挂在该表单下
provideFormName(computed(() => p.props.name));
</script>

<template>
  <el-card shadow="never" class="openui-form">
    <NodeSlot :value="p.props.fields" :render-node="renderNode" />
    <div class="form-actions">
      <NodeSlot :value="p.props.buttons" :render-node="renderNode" />
    </div>
  </el-card>
</template>

<style scoped>
.openui-form :deep(.el-card__body) {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.form-actions {
  display: flex;
  gap: 12px;
  justify-content: flex-end;
}
</style>
