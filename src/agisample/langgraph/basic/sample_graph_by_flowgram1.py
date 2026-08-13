import json
import operator
import os
from pathlib import Path
from typing import Dict, Any

import requests
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict, Annotated

from langgraph.graph import StateGraph


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class HttpNodeHandlerFactory:

    @staticmethod
    def _render_template(template_str: str, context: Dict[str, Any]) -> str:
        """
        简单的模板渲染引擎。
        支持将 "{user_id}" 替换为 context["user_id"] 的值。
        生产环境建议使用 Jinja2 以支持更复杂的逻辑。
        """
        if not isinstance(template_str, str):
            return template_str

        try:
            # 使用 python 内置的 format 方法进行简单的变量替换
            # 注意：这要求 context 中的 key 必须与 template 中的占位符完全匹配
            return template_str.format(**context)
        except KeyError as e:
            # 如果找不到变量，可以选择报错或者保留原样
            print(f"Warning: Variable {e} not found in context.")
            return template_str
        except Exception as e:
            print(f"Template render error: {e}")
            return template_str


    @staticmethod
    def create_node(config):

        def _execute(state: State):
            # 1. 从 state 中获取上下文
            context = state["context"]

            url = config.get("url")
            method = config.get("method", "GET").upper()

            headers = config.get("headers", {})
            for k, v in headers.items():
                headers[k] = HttpNodeHandlerFactory._render_template(v, context)

            json_body = None
            raw_body = config.get("body")
            if raw_body:
                # 简单处理：如果是字典，尝试替换 value 中的变量
                if isinstance(raw_body, dict):
                    # 深拷贝以防修改原配置，这里做简化的单层替换演示
                    json_body = {}
                    for k, v in raw_body.items():
                        json_body[k] = HttpNodeHandlerFactory._render_template(v, context)
                else:
                    json_body = raw_body


            params = config.get("params")
            body = config.get("body")

            # 2. 发起请求
            print(f"--- Executing HTTP Node: {method} {url} ---")
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_body,
                    timeout=config.get("timeout", 10)
                )
                response.raise_for_status()  # 如果状态码不是 2xx 则抛出异常

                # 尝试解析 JSON，如果不是 JSON 则取 text
                try:
                    result_data = response.json()
                except:
                    result_data = response.text

                status_code = response.status_code

            except Exception as e:
                print(f"HTTP Request Failed: {e}")
                result_data = {"error": str(e)}
                status_code = 500

            # 3. 更新 State
            # 将结果写入指定的 output_key，如果没有指定，默认写入 http_last_response
            output_key = config.get("output_key", "http_last_response")

            # 我们通常不仅仅保存数据，还保存状态码，以便后续节点做路由判断
            state["context"][output_key] = {
                "status": status_code,
                "data": result_data
            }

            return state

        return _execute


class DynamicGraphBuilder:
    def __init__(self, workflow_json):
        self.config = workflow_json
        self.workflow = StateGraph(State)

    def build(self):
        nodes = self.config["nodes"]
        edges = self.config["edges"]

        # 1. 添加节点
        for node in nodes:
            node_id = node["id"]
            node_type = node["type"]
            node_config = node["data"]

            if node_type == "start":
                # Start 节点通常是入口，不做具体逻辑，或者初始化
                continue
            elif node_type == "http":
                handler = HttpNodeHandlerFactory.create_node(node_config)
                self.workflow.add_node(node_id, handler)
            # ... 处理其他类型

        # 2. 添加边 (Edges)
        for edge in edges:
            src = edge["sourceNodeID"]
            tgt = edge["targetNodeID"]

            # 处理条件边 (Conditional Edges)
            if "condition" in edge:
                # 这里的逻辑比较复杂，需要将同一个 source 的所有条件边收集起来
                # 构建一个路由函数 (router function)
                pass
            else:
                # 普通边
                # 注意：如果 source 是 start，则设为 entry_point
                if self.get_node_type(src) == "start":
                    self.workflow.set_entry_point(tgt)
                else:
                    self.workflow.add_edge(src, tgt)

        return self.workflow.compile()

    def get_node_type(self, node_id):
        # 辅助函数：根据 ID 查找类型
        for n in self.config["nodes"]:
            if n["id"] == node_id:
                return n["type"]
        return None


path = Path(os.path.join(Path(__file__).resolve().parents[4], "data", "flow.json"))
with path.open("r", encoding="utf-8") as f:
    workflow_json = json.load(f)

print(workflow_json)

builder = DynamicGraphBuilder(workflow_json)
app = builder.build()

initial_state = {"context": {"url": ""}, "messages": []}
result = app.invoke(initial_state)

print(result)
