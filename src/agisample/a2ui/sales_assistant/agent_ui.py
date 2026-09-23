"""模型输出的 Schema、组件树、数据来源和点击事件校验。"""
import json
from pathlib import Path
import re
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from .protocol import CATALOG_ID, VERSION
from .service import validate_filters

ROOT = Path(__file__).parent


def resolve(value, model, scope=""):
    if not isinstance(value, dict):
        return value
    if set(value) != {"path"} or not isinstance(value["path"], str):
        raise ValueError("数据绑定必须是单个 path。")
    path = value["path"]
    if not path.startswith("/"):
        if not scope:
            raise ValueError("非表格行只支持绝对路径。")
        path = scope + "/" + path
    if not path.startswith("/queries/"):
        raise ValueError("数据只能绑定到本轮查询工具结果。")
    node = model
    for key in path[1:].split("/"):
        key = key.replace("~1", "/").replace("~0", "~")
        if key in ("__proto__", "constructor", "prototype"):
            raise ValueError("不安全的数据路径。")
        try:
            node = node[int(key)] if isinstance(node, list) and key.isdecimal() else node[key]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ValueError("绑定路径不存在：" + path) from None
    return node


def event_context(event, model, scope=""):
    name = event.get("name")
    if name not in ("filter_region", "show_details", "show_summary"):
        raise ValueError("不支持的界面事件。")
    raw = event.get("context")
    expected = {"period", "region", "view"} if name == "filter_region" else {"period", "region"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("事件 context 字段不完整或含额外字段。")
    values = {key: resolve(value, model, scope) for key, value in raw.items()}
    view = values.get("view", "details" if name == "show_details" else "summary")
    validate_filters(values["period"], values["region"], view)
    return values


class UIValidator:
    def __init__(self):
        def read(path):
            return json.loads(path.read_text(encoding="utf-8-sig"))
        common = read(ROOT / "schemas/common_types.json")
        catalog = read(ROOT / "web/catalog.json")
        server = read(ROOT / "schemas/server_to_client.json")
        client = read(ROOT / "schemas/client_to_server.json")
        registry = Registry().with_resources([
            (common["$id"], Resource.from_contents(common)),
            (catalog["$id"], Resource.from_contents(catalog)),
            ("https://a2ui.org/specification/v0_9/catalog.json", Resource.from_contents(catalog))])
        self.server = Draft202012Validator(server, registry=registry)
        self.client = Draft202012Validator(client, format_checker=FormatChecker())

    def parse(self, content, model, surface_id):
        if not isinstance(content, str) or len(content) > 100_000:
            raise ValueError("最终响应必须是大小受限的 JSON 对象。")
        try:
            output = json.loads(content)
        except ValueError:
            raise ValueError("最终响应必须是 JSON，不要使用代码围栏。") from None
        if not isinstance(output, dict) or set(output) != {"reply", "components"}:
            raise ValueError("最终输出仅包含 reply 和 components。")
        reply, components = output["reply"], output["components"]
        if not isinstance(reply, str) or not 1 <= len(reply) <= 1000:
            raise ValueError("reply 必须是简短中文答复。")
        if re.search(r"[0-9¥￥%]", reply):
            raise ValueError("reply 不得写数值结论；请用组件绑定工具数据。")
        if not isinstance(components, list) or not 1 <= len(components) <= 80:
            raise ValueError("组件数量必须在一到八十之间。")
        envelope = {"version": VERSION, "updateComponents": {"surfaceId": surface_id, "components": components}}
        if list(self.server.iter_errors(envelope)):
            raise ValueError("组件 Schema 校验失败，请检查目录支持的字段和必填属性。")
        by_id = {}
        for component in components:
            identifier = component["id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier) or identifier in by_id:
                raise ValueError("组件 ID 必须唯一且由字母、数字、下划线、短横线组成。")
            by_id[identifier] = component
        visited = set()
        def walk(identifier, ancestors):
            if identifier not in by_id:
                raise ValueError("组件引用不存在：" + identifier)
            if identifier in ancestors or len(ancestors) > 35:
                raise ValueError("组件树存在循环或嵌套过深。")
            if identifier in visited:
                raise ValueError("同一组件不能被重复引用，请为不同位置使用独立 ID。")
            visited.add(identifier)
            component = by_id[identifier]
            for child in component.get("children", []) + ([component["child"]] if "child" in component else []):
                walk(child, ancestors | {identifier})
        walk("root", set())
        if visited != set(by_id):
            raise ValueError("组件必须全部能从 root 到达。")
        for component in components:
            kind = component["component"]
            if kind == "Text":
                text = component["text"]
                if isinstance(text, str):
                    if re.search(r"[0-9¥￥%]", text):
                        raise ValueError("数值/日期/金额必须绑定查询结果，不得写死在 Text 中。")
                elif not isinstance(resolve(text, model), (str, int, float)):
                    raise ValueError("Text 必须绑定标量数据。")
            elif kind == "Button":
                if by_id[component["child"]]["component"] != "Text":
                    raise ValueError("Button 的 child 必须是 Text。")
                event_context(component["action"]["event"], model)
            elif kind == "DataTable":
                path = component["rows"]["path"]
                if not re.fullmatch(r"/queries/q[1-4]/rows", path):
                    raise ValueError("表格必须绑定本轮工具返回的 rows。")
                rows = resolve(component["rows"], model)
                query = model["queries"][path.split("/")[2]]
                if component["caption"] != {"path": path.removesuffix("/rows") + "/title"}:
                    raise ValueError("表格 caption 必须绑定对应查询的 title，保留查询口径。")
                if not component["columns"] or any(column not in query["columns"] for column in component["columns"]):
                    raise ValueError("表格列必须从工具返回的 columns 原样选择，不得篡改标签或单位。")
                if len({c["key"] for c in component["columns"]}) != len(component["columns"]):
                    raise ValueError("表格列不能重复。")
                if component.get("rowAction"):
                    for index in range(len(rows)):
                        event_context(component["rowAction"]["event"], model, path + "/" + str(index))
                    if component["rowAction"]["event"]["name"] not in ("show_details", "show_summary", "filter_region"):
                        raise ValueError("表格行事件不受支持。")
        return reply, components

    def validate_action(self, message, surface_id, components, model):
        if list(self.client.iter_errors(message)) or message.get("version") != VERSION:
            raise ValueError("无效的 A2UI action 消息。")
        event = message.get("action", {})
        if event.get("surfaceId") != surface_id:
            raise ValueError("操作不属于当前会话界面。")
        component = next((c for c in components if c["id"] == event["sourceComponentId"]), None)
        if not component:
            raise ValueError("事件来源组件不存在或已经失效。")
        candidates = []
        if component["component"] == "Button":
            definition = component["action"]["event"]
            candidates = [(definition["name"], event_context(definition, model))]
        elif component["component"] == "DataTable" and component.get("rowAction"):
            definition = component["rowAction"]["event"]
            rows = resolve(component["rows"], model)
            candidates = [(definition["name"], event_context(definition, model, component["rows"]["path"] + "/" + str(i)))
                          for i in range(len(rows))]
        if (event["name"], event["context"]) not in candidates:
            raise ValueError("事件与当前界面发布的操作不匹配，请重新查询。")
        return event

    def messages(self, surface_id, data, components, initial):
        def envelope(kind, **fields):
            return {"version": VERSION, kind: {"surfaceId": surface_id, **fields}}
        messages = [] if initial else [envelope("deleteSurface")]
        messages += [envelope("createSurface", catalogId=CATALOG_ID),
                     envelope("updateDataModel", path="/", value=data),
                     envelope("updateComponents", components=components)]
        for message in messages:
            if list(self.server.iter_errors(message)):
                raise ValueError("生成的 A2UI 消息未通过协议校验。")
        return messages
