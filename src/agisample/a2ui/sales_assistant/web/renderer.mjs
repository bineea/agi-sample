import {resolve, actionMessage} from "./state.mjs";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text ?? "";
  return node;
}

export function renderSurface(surfaceId, surface, onAction) {
  const {components, model} = surface;
  const value = (v, scope = "") => resolve(v, model, scope);
  function dispatch(event, componentId, scope = "") {
    onAction(actionMessage(event, surfaceId, componentId, model, scope));
  }
  function render(id, ancestors = new Set()) {
    if (ancestors.has(id) || ancestors.size > 40) throw new Error("组件树存在循环或嵌套过深。");
    const c = components.get(id);
    if (!c) return element("span", "placeholder");
    const next = new Set(ancestors).add(id);
    let node;
    switch (c.component) {
      case "Text": {
        const variants = {h2: ["h2", "result-title"], metric: ["p", "metric-value"],
          muted: ["p", "muted"], body: ["span", "body-text"]};
        const [tag, className] = variants[c.variant] ?? variants.body;
        node = element(tag, className, value(c.text));
        break;
      }
      case "Row":
      case "Column":
        node = element("div", c.component === "Row" ? "a2ui-row" : "a2ui-column");
        for (const child of c.children) node.append(render(child, next));
        break;
      case "Card":
        node = element("section", "metric-card");
        node.append(render(c.child, next));
        break;
      case "Button":
        node = element("button", c.variant === "primary" ? "a2ui-button selected" : "a2ui-button");
        node.type = "button";
        node.setAttribute("aria-pressed", String(c.variant === "primary"));
        node.append(render(c.child, next));
        node.addEventListener("click", () => dispatch(c.action.event, c.id));
        break;
      case "DataTable": {
        node = element("div", "table-scroll");
        node.tabIndex = 0;
        node.setAttribute("aria-label", "查询结果表格，可横向滚动");
        const table = element("table");
        table.append(element("caption", "sr-only", value(c.caption)));
        const head = element("thead"), heading = element("tr");
        for (const column of c.columns) {
          const th = element("th", column.align === "right" ? "numeric" : "", column.label);
          th.scope = "col";
          heading.append(th);
        }
        if (c.rowAction) heading.append(element("th", "", "操作"));
        head.append(heading); table.append(head);
        const body = element("tbody"), rows = value(c.rows) ?? [];
        if (!Array.isArray(rows)) throw new Error("表格数据必须是数组。");
        for (const [index, row] of rows.entries()) {
          const tr = element("tr");
          for (const column of c.columns)
            tr.append(element("td", column.align === "right" ? "numeric" : "", row[column.key]));
          if (c.rowAction) {
            const cell = element("td");
            const button = element("button", "table-action", c.rowAction.label);
            button.type = "button";
            button.addEventListener("click", () => dispatch(c.rowAction.event, c.id, c.rows.path + "/" + index));
            cell.append(button); tr.append(cell);
          }
          body.append(tr);
        }
        if (!rows.length) {
          const tr = element("tr"), td = element("td", "empty-cell", c.emptyText);
          td.colSpan = c.columns.length + (c.rowAction ? 1 : 0);
          tr.append(td); body.append(tr);
        }
        table.append(body); node.append(table);
        break;
      }
      default: throw new Error("不支持的组件类型。");
    }
    node.dataset.componentId = c.id;
    return node;
  }
  return render("root");
}
