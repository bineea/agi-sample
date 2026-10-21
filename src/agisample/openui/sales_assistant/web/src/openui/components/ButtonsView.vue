<script setup lang="ts">
import type { ComponentRenderProps } from "@openuidev/vue-lang";
import { NodeSlot } from "../NodeSlot";

defineProps<ComponentRenderProps<{ buttons: unknown }>>();
</script>

<template>
  <div class="openui-buttons">
    <NodeSlot :value="props.buttons" :render-node="renderNode" />
  </div>
</template>

<style scoped>
.openui-buttons {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
</style>
