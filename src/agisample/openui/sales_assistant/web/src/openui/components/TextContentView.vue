<script setup lang="ts">
import type { ComponentRenderProps } from "@openuidev/vue-lang";

defineProps<ComponentRenderProps<{ text: string }>>();
</script>

<template>
  <p class="openui-text">{{ props.text }}</p>
</template>

<style scoped>
.openui-text {
  margin: 0;
  font-size: 15px;
  line-height: 1.6;
}
</style>
