import { computed, onBeforeUnmount, ref, type ComputedRef } from "vue";
import { useFormName, useOpenUI } from "@openuidev/vue-lang";
import { evaluate, isReactiveAssign } from "@openuidev/lang-core";

/**
 * Vue 版 useStateField（对齐官方 react-ui 的字段行为）：
 * - 若字段值是 $variable 绑定（ReactiveAssign），读写直接落到共享 store，
 *   实现 openui-lang 的双向绑定语义；
 * - 否则读写 Renderer 的表单字段状态（onStateUpdate 的来源）。
 */
export function useStateField(
  name: () => string,
  bindingValue: () => unknown,
  componentType: string,
) {
  const ctx = useOpenUI();
  const formNameRef = useFormName();

  // store 没有 Vue 响应式集成，订阅后手动触发重算
  const tick = ref(0);
  const unsubscribe = ctx.store.subscribe(() => {
    tick.value++;
  });
  onBeforeUnmount(unsubscribe);

  const formName: ComputedRef<string | undefined> = computed(
    () => formNameRef?.value,
  );

  const value = computed<unknown>(() => {
    void tick.value;
    const binding = bindingValue();
    if (isReactiveAssign(binding)) {
      return ctx.store.get(binding.target);
    }
    return ctx.getFieldValue(formName.value, name()) ?? binding;
  });

  function setValue(v: unknown, save = true) {
    const binding = bindingValue();
    if (isReactiveAssign(binding)) {
      // 对齐 resolveStateField：写回时先对绑定表达式求值（$value 为用户输入）
      const next = evaluate(binding.expr, {
        ...ctx.evaluationContext,
        extraScope: { $value: v },
      } as any);
      ctx.store.set(binding.target, next);
      return;
    }
    ctx.setFieldValue(formName.value, componentType, name(), v, save);
  }

  return { value, setValue, formName };
}
