import { defineComponent, h, type PropType } from "vue";
import type { RenderNodeResult } from "@openuidev/vue-lang";

/**
 * 渲染插槽：把 renderNode(value) 的结果（VNode / 数组 / 字符串）渲染到模板中。
 * vue-lang 的 renderNode 通过 props 传入，模板无法直接渲染 VNode 数组，
 * 用这个函数式组件包一层即可：<NodeSlot :value="props.children" :render-node="renderNode" />
 */
export const NodeSlot = defineComponent({
  name: "NodeSlot",
  props: {
    value: { type: null as unknown as PropType<unknown>, default: null },
    renderNode: {
      type: Function as PropType<(value: unknown) => RenderNodeResult>,
      required: true,
    },
  },
  setup(p) {
    return () => {
      const out = p.renderNode(p.value);
      if (out == null) return null;
      // 字符串包成文本 VNode，数组/单个 VNode 直接返回
      return typeof out === "string" ? h("span", out) : (out as any);
    };
  },
});
