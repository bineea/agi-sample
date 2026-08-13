import json
import operator
import logging
import asyncio
import os
from io import BytesIO
from pathlib import Path

import httpx
from typing import Dict, Any, List, TypedDict, Annotated, Callable, Union, Optional

from PIL import Image
from jinja2 import Template
from langgraph.graph import StateGraph, END, START

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# 1. 定义动态状态 (Dynamic State)
# -------------------------------------------------------------------------
# 这是一个通用的状态容器，用于模拟 Flowgram 中的上下文变量
class WorkflowState(TypedDict):
    # 存储所有流程变量的扁平化字典
    variables: Dict[str, Any]
    # 可选：存储消息历史（如果涉及对话）
    messages: Annotated[List[str], operator.add]


# -------------------------------------------------------------------------
# 2. 核心工具函数 (Template Engine & Executors)
# -------------------------------------------------------------------------

class DataResolver:
    """解析 value 配置，处理 ref (引用) 和 constant (常量)"""

    @staticmethod
    def resolve(value_config: dict, state: WorkflowState):
        if not isinstance(value_config, dict):
            return value_config

        v_type = value_config.get("type")

        # 处理常量
        if v_type == "constant":
            return value_config.get("content")

        # 处理引用 (Ref) ["node_id", "property_name"]
        elif v_type == "ref":
            content = value_config.get("content", [])
            return DataResolver._resolve_ref(content, state)

        # 递归处理对象或表达式 (简化版)
        return value_config

    @staticmethod
    def _resolve_ref(content: Any, state: WorkflowState):
        variables = state.get("variables", {}) if isinstance(state, dict) else {}
        if not variables:
            return None

        if isinstance(content, (list, tuple)):
            # 顶层优先级：尝试将 node_id 作为命名空间
            parts = [str(part) for part in content if part is not None]
            if not parts:
                return None

            node_id, *path = parts
            current = variables.get(node_id)
            for key in path:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    current = None
                if current is None:
                    break
            if current is not None:
                return current

            # 回退 1：直接使用末尾字段名（兼容扁平变量存储）
            if path:
                fallback_key = path[-1]
                if fallback_key in variables:
                    return variables.get(fallback_key)

            # 回退 2：将路径以 . 连接作为 key
            dotted_key = ".".join(parts)
            if dotted_key in variables:
                return variables[dotted_key]

        return None


class LogicEvaluator:
    """解析分支节点的条件逻辑"""

    @staticmethod
    def evaluate(condition: dict, state: WorkflowState) -> bool:
        value_block = condition.get("value", {}) if isinstance(condition, dict) else {}
        left_config = value_block.get("left")
        right_config = value_block.get("right")
        operator_key = value_block.get("operator")

        left_val = DataResolver.resolve(left_config, state)
        right_val = DataResolver.resolve(right_config, state)

        if operator_key == "is_true":
            return bool(left_val) is True
        if operator_key == "is_not_empty":
            return left_val not in (None, "", [], {})
        if operator_key == "equals":
            return left_val == right_val

        return False

def render_template(template_str: Union[str, Any], variables: Dict[str, Any]) -> Any:
    """
    使用 Jinja2 渲染字符串中的 {{ variable }} 占位符。
    如果输入不是字符串，则原样返回。
    """
    if isinstance(template_str, str) and "{{" in template_str:
        try:
            # 允许在模板中使用 python 内置方法，如.upper()
            return Template(template_str).render(**variables)
        except Exception as e:
            logger.error(f"Template rendering failed for '{template_str}': {e}")
            return template_str
    # 如果是字典或列表，递归渲染
    elif isinstance(template_str, dict):
        return {k: render_template(v, variables) for k, v in template_str.items()}
    elif isinstance(template_str, list):
        return [render_template(item, variables) for item in template_str]
    return template_str


# -------------------------------------------------------------------------
# 3. 节点工厂 (Node Factory)
# -------------------------------------------------------------------------

class NodeFactory:
    """根据 JSON 配置生成对应的 LangGraph 节点函数"""

    @staticmethod
    def create_start_node(node_config: Dict[str, Any]):
        """处理开始节点，初始化变量"""

        def start_node(state: WorkflowState):
            # 获取初始输入并更新到变量池
            initial_inputs = node_config.get("data", {}).get("inputs", {})
            current_vars = state.get("variables", {}).copy()
            current_vars.update(initial_inputs)
            logger.info(f"🚀 Workflow initiated. Init vars: {initial_inputs.keys()}")
            return {"variables": current_vars}

        return start_node

    @staticmethod
    def create_http_node(node_config: Dict[str, Any]):
        """
        动态生成 HTTP 请求节点
        支持：Method, URL, Headers, Body, Timeout
        """
        node_id = node_config["id"]
        data = node_config.get("data", {})

        async def http_node(state: WorkflowState):
            variables = state.get("variables", {})

            # 1. 动态渲染配置参数
            url = render_template(data.get("url", ""), variables)
            method = data.get("method", "GET").upper()
            headers = render_template(data.get("headers", {}), variables)
            body = render_template(data.get("body", {}), variables)
            timeout = data.get("timeout", 10)

            logger.info(f"🌐 {method} {url}")

            # 2. 执行请求
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=body if method in ['POST','PUT','PATCH','DELETE'] else None,
                        params=body if method == "GET" else None,
                        timeout=timeout
                    )
                    response.raise_for_status()
                    result = response.json()
                    status = "success"
                except Exception as e:
                    logger.error(f"HTTP Request failed: {e}")
                    result = {"error": str(e)}
                    status = "error"

            # 3. 输出处理 (默认写入 node_id 为 key 的变量中)
            output_key = data.get("output_var", f"{node_id}_result")
            new_vars = variables.copy()
            new_vars[output_key] = {
                "status": status,
                "status_code": response.status_code if 'response' in locals() else 0,
                "data": result
            }
            return {"variables": new_vars}

        return http_node

    @staticmethod
    def create_code_node(node_config: Dict[str, Any]):
        """
        处理 Python 代码执行节点
        警告：exec() 存在安全风险，仅在受控环境中使用
        """
        node_id = node_config["id"]
        code_script = node_config.get("data", {}).get("code", "")

        def code_node(state: WorkflowState):
            variables = state.get("variables", {}).copy()
            logger.info(f"💻 [Code] Executing script for {node_id}")

            try:
                # 创建安全的执行上下文
                local_scope = {"inputs": variables, "outputs": {}}
                exec(code_script, {}, local_scope)

                # 获取 outputs 并合并回主变量池
                script_outputs = local_scope.get("outputs", {})
                if isinstance(script_outputs, dict):
                    variables.update(script_outputs)
            except Exception as e:
                logger.error(f"Code execution error: {e}")
                variables[f"{node_id}_error"] = str(e)

            return {"variables": variables}

        return code_node

    @staticmethod
    def create_router(node_config: Dict[str, Any], edges: List):
        """
        为分支节点创建路由逻辑
        """
        branches = node_config.get("data", {}).get("branches", [])

        def router_function(state: WorkflowState):
            variables = state.get("variables", {})

            # 遍历定义的分支条件
            for branch in branches:
                logic = branch.get("logic", "or")  # 默认使用 "or" 逻辑
                conditions = branch.get("conditions", [])
                
                # 评估所有条件
                results = [LogicEvaluator.evaluate(c, state) for c in conditions]
                
                # 根据逻辑类型组合结果
                is_pass = False
                if logic == "or":
                    is_pass = any(results)
                else:
                    is_pass = all(results)
                
                target_node_id = None

                # 查找该分支对应的目标节点 ID (从 edges 中查找)
                branch_id = branch.get("id")
                for edge in edges:
                    if edge["sourceNodeID"] == node_config["id"] and edge.get("sourcePortID") == branch_id:
                        target_node_id = edge["targetNodeID"]
                        break

                if not target_node_id:
                    continue

                if is_pass:
                    logger.info(f"🔀 Matched condition: {branch_id} -> {target_node_id}")
                    return target_node_id

            # 默认 fallback 到 else 分支或结束
            # 查找 default handle
            for edge in edges:
                if edge["sourceNodeID"] == node_config["id"] and edge.get("sourcePortID") == "else":
                    return edge["targetNodeID"]

            return END

        return router_function


# -------------------------------------------------------------------------
# 4. 图构建器 (Graph Builder)
# -------------------------------------------------------------------------

class FlowgramCompiler:
    def __init__(self, flow_json: Dict[str, Any]):
        self.json_data = flow_json
        self.workflow = StateGraph(WorkflowState)
        self.factory = NodeFactory()

    def compile(self):
        nodes = self.json_data.get("nodes", [])
        edges = self.json_data.get("edges", [])

        # 1. 添加所有节点
        for node in nodes:
            node_type = node.get("type")
            node_id = node["id"]

            if node_type == "start":
                self.workflow.add_node(node_id, self.factory.create_start_node(node))
                self.workflow.set_entry_point(node_id)

            elif node_type == "http":
                self.workflow.add_node(node_id, self.factory.create_http_node(node))

            elif node_type == "code":
                self.workflow.add_node(node_id, self.factory.create_code_node(node))

            # Branch 节点通常不是执行节点，而是路由逻辑
            # 我们在这里不添加它为 node，就是在处理 edge 时作为条件边处理
            # 或者将其添加为一个空操作节点(Passthrough)，然后接条件边
            elif node_type == "branches":
                # 添加一个空节点作为锚点
                self.workflow.add_node(node_id, lambda state: state)
            
            # 添加对 end 类型节点的处理
            elif node_type == "end":
                # 添加一个空节点作为结束节点
                self.workflow.add_node(node_id, lambda state: state)

        # 2. 添加所有边
        for node in nodes:
            node_id = node["id"]
            node_type = node.get("type")

            if node_type == "branches":
                # 处理条件分支逻辑
                router = self.factory.create_router(node, edges)

                # 收集所有可能的下游节点作为 path_map
                destinations = {}
                for edge in edges:
                    if edge["sourceNodeID"] == node_id:
                        destinations[edge["targetNodeID"]] = edge["targetNodeID"]
                
                # 添加 END 作为有效目标
                destinations[END] = END
                
                # 添加条件边
                if destinations:
                    self.workflow.add_conditional_edges(
                        node_id,
                        router,
                        destinations
                    )
            else:
                # 处理普通边
                # 找到当前节点的所有出边
                outgoing_edges = [e for e in edges if e["sourceNodeID"] == node_id]
                for edge in outgoing_edges:
                    self.workflow.add_edge(node_id, edge["targetNodeID"])

        return self.workflow.compile()


# -------------------------------------------------------------------------
# 5. 模拟运行 (Demo)
# -------------------------------------------------------------------------

async def main():
    # 读取flow.json
    path = Path(os.path.join(Path(__file__).resolve().parents[4], "data", "flow.json"))
    with path.open("r", encoding="utf-8") as f:
        workflow_json = json.load(f)

    print("🔧 Compiling Flowgram JSON to LangGraph...")
    compiler = FlowgramCompiler(workflow_json)
    app = compiler.compile()

    try:
        image_obj = app.get_graph(xray=True).draw_mermaid_png()
        image = Image.open(BytesIO(image_obj))
        image.show()
        # with open('app.png', 'wb') as f:
        #     f.write(image_obj)
    except Exception as e:
        logger.error(f"Failed to generate graph image: {e}")
        pass


    print("🏃 Running Workflow...")
    # 初始化状态
    initial_state = {"variables": {"start_0": {"enable": True}}, "messages":[]}

    # 执行图
    result = await app.ainvoke(initial_state)

    print("\n✅ Execution Finished!")
    print("Final Variables:", json.dumps(result["variables"], indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())