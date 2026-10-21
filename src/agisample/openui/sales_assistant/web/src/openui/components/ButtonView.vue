<script setup lang="ts">
import {
  useFormName,
  useIsStreaming,
  useTriggerAction,
  type ComponentRenderProps,
} from "@openuidev/vue-lang";

const p = defineProps<
  ComponentRenderProps<{ label: string; action?: unknown; variant?: string }>
>();

const triggerAction = useTriggerAction();
const formName = useFormName();
const isStreaming = useIsStreaming();

function onClick() {
  // action 是 Action([...]) 求值后的 ActionPlan（{steps: [...]}），
  // triggerAction 会按步骤执行：@Run / @ToAssistant / @OpenUrl / @Set / @Reset
  triggerAction(p.props.label, formName?.value, p.props.action as any);
}
</script>

<template>
  <el-button
    :type="p.props.variant === 'primary' ? 'primary' : 'default'"
    :disabled="isStreaming"
    @click="onClick"
  >
    {{ p.props.label }}
  </el-button>
</template>
