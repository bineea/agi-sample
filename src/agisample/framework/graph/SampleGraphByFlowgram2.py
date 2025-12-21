import json
import operator
from typing import Annotated, Any, Dict, List, TypedDict, Union, Callable
from langgraph.graph import StateGraph, END

# 定义全局状态
class WorkflowState(TypedDict):
    # 存储每个节点的输出结果
    # Key: node_id, Value: 节点的输出字典
    node_outputs: Dict[str, Any]
    # 初始输入参数
    inputs: Dict[str, Any]


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
            if len(content) == 2:
                ref_node_id, ref_key = content
                # 从全局状态中查找上游节点的输出
                node_data = state["node_outputs"].get(ref_node_id, {})
                return node_data.get(ref_key)

        # 递归处理对象或表达式 (简化版)
        return value_config


class LogicEvaluator:
    """解析分支节点的条件逻辑"""

    @staticmethod
    def evaluate(condition: dict, state: WorkflowState) -> bool:
        # 获取左值 (通常是 ref)
        left_config = condition.get("value", {}).get("left")
        left_val = DataResolver.resolve(left_config, state)

        op = condition.get("value", {}).get("operator")

        # 实现简单的算子
        if op == "is_true":
            return bool(left_val) is True
        elif op == "is_not_empty":
            return left_val is not None and left_val != "" and left_val != []
        elif op == "equals":
            # 需解析右值，这里省略
            return True

        return False


class NodeHandlers:

    @staticmethod
    def handle_start(node_config, state: WorkflowState):
        """开始节点：将初始 inputs 映射到 outputs"""
        # Start 节点通常直接透传初始输入，或者根据 default 值初始化
        node_id = node_config["id"]
        # 简单起见，直接把全局 inputs 视为 start 节点的输出
        return {node_id: state["inputs"]}

    @staticmethod
    def handle_code(node_config, state: WorkflowState):
        """代码节点：执行脚本"""
        node_id = node_config["id"]
        inputs_def = node_config["data"].get("inputsValues", {})

        # 1. 解析输入参数
        func_inputs = {}
        for key, conf in inputs_def.items():
            func_inputs[key] = DataResolver.resolve(conf, state)

        print(f"--- [Code Node {node_id}] Executing with inputs: {func_inputs} ---")

        # 2. 执行逻辑
        # 注意：你的 JSON 是 Java 代码，Python 无法直接执行。
        # 这里模拟一个 Python 版本的逻辑：返回 inputs + "helloCodeExecute": "123"
        result = func_inputs.copy()
        result["helloCodeExecute"] = "123 (Mocked Java Result)"

        return {node_id: result}

    @staticmethod
    def handle_end(node_config, state: WorkflowState):
        """结束节点：组装最终结果"""
        node_id = node_config["id"]
        inputs_def = node_config["data"].get("inputsValues", {})

        final_output = {}
        for key, conf in inputs_def.items():
            final_output[key] = DataResolver.resolve(conf, state)

        print(f"--- [End Node] Final Output: {final_output} ---")
        return {node_id: final_output, "_final_result": final_output}

    @staticmethod
    def evaluate_branch(node_config, state: WorkflowState):
        """分支节点：纯逻辑判断，不产生数据，返回路径 ID"""
        branches = node_config["data"].get("branches", [])

        for branch in branches:
            logic = branch.get("logic", "or")  # and / or
            conditions = branch.get("conditions", [])
            results = [LogicEvaluator.evaluate(c, state) for c in conditions]

            is_pass = False
            if logic == "or":
                is_pass = any(results)
            else:
                is_pass = all(results)

            if is_pass:
                print(f"--- [Branch Node] Condition met: {branch['id']} ---")
                return branch["id"]  # 返回满足条件的端口 ID

        print(f"--- [Branch Node] No condition met, going ELSE ---")
        return "else"


class GraphCompiler:
    def __init__(self, flow_json: Dict[str, Any]):
        self.nodes_cfg = {n["id"]: n for n in flow_json["nodes"]}
        self.edges_cfg = flow_json["edges"]
        self.workflow = StateGraph(WorkflowState)

    def _make_node_func(self, node_cfg):
        """闭包工厂：创建一个符合 LangGraph 签名的节点函数"""
        node_type = node_cfg["type"]

        def _func(state: WorkflowState):
            new_data = {}
            if node_type == "start":
                new_data = NodeHandlers.handle_start(node_cfg, state)
            elif node_type == "code":
                new_data = NodeHandlers.handle_code(node_cfg, state)
            elif node_type == "end":
                new_data = NodeHandlers.handle_end(node_cfg, state)

            # 更新全局状态中的 node_outputs
            # 注意：LangGraph 的状态更新是增量的 (Dependency specific)
            # 这里我们手动合并字典
            current_outputs = state.get("node_outputs", {}).copy()
            current_outputs.update(new_data)
            return {"node_outputs": current_outputs}

        return _func

    def _make_router_func(self, node_cfg, edge_map):
        """创建路由函数"""

        def _router(state: WorkflowState):
            # 1. 计算走哪个分支
            port_id = NodeHandlers.evaluate_branch(node_cfg, state)
            # 2. 查找该端口对应的目标节点 ID
            target_node = edge_map.get(port_id)
            if not target_node:
                # 如果没有连线，通常结束或报错
                return END
            return target_node

        return _router

    def compile(self):
        # 1. 添加节点
        for node_id, node_cfg in self.nodes_cfg.items():
            node_type = node_cfg["type"]

            if node_type == "branches":
                # 分支节点在 LangGraph 中通常体现为 Conditional Edge，
                # 但为了可视化一致性，我们也可以把它当做一个不做操作的节点，
                # 或者直接在后续处理 Edge 时处理它。
                # 策略：不添加物理节点，而是将其作为路由逻辑的锚点。
                pass
            else:
                self.workflow.add_node(node_id, self._make_node_func(node_cfg))

        # 2. 处理入口
        start_nodes = [n for n in self.nodes_cfg.values() if n["type"] == "start"]
        if start_nodes:
            self.workflow.set_entry_point(start_nodes[0]["id"])

        # 3. 添加边 (Edges)
        # 预处理：将边按 Source 分组
        edges_by_source = {}
        for edge in self.edges_cfg:
            src = edge["sourceNodeID"]
            if src not in edges_by_source:
                edges_by_source[src] = []
            edges_by_source[src].append(edge)

        for src_id, edges in edges_by_source.items():
            src_node = self.nodes_cfg[src_id]

            # 情况 A: 源节点是普通节点 -> 直接连线
            if src_node["type"] != "branches":
                target_id = edges[0]["targetNodeID"]
                # 如果目标是分支节点，我们需要“透传”连接到分支逻辑
                # 但 LangGraph 可以在 add_conditional_edges 时指定 source

                target_node = self.nodes_cfg[target_id]

                if target_node["type"] == "branches":
                    # 如果下一跳是分支节点，我们需要在这里定义路由逻辑
                    # 构建路由映射表: { "branch_id": "target_node_id" }
                    branch_edges = edges_by_source.get(target_id, [])
                    route_map = {}
                    for be in branch_edges:
                        port = be.get("sourcePortID", "else")
                        route_map[port] = be["targetNodeID"]

                    # 添加条件边：从 src_id 出发，经过 target_id(逻辑)，去往不同节点
                    self.workflow.add_conditional_edges(
                        src_id,
                        self._make_router_func(target_node, route_map),
                        route_map
                    )
                else:
                    self.workflow.add_edge(src_id, target_id)

            # 情况 B: 源节点是分支节点
            # (已经在上方作为目标节点处理过了，跳过)
            pass

        return self.workflow.compile()


if __name__ == "__main__":
    # 加载你的 JSON (这里假设 content 是你提供的 JSON 字符串或对象)
    # import json
    # flow_data = json.loads(your_json_string)

    # 为方便演示，直接使用你提供的结构
    flow_data = {
        "nodes": [
            {"id": "start_0", "type": "start",
             "data": {"outputs": {"properties": {"query": {"default": "Hello"}, "enable": {"default": True}}}}},
            {"id": "end_0", "type": "end",
             "data": {"inputsValues": {"query": {"type": "ref", "content": ["start_0", "query"]}}}},
            {"id": "condition_alRhO", "type": "branches", "data": {"branches": [{"id": "branch_R_7LT", "logic": "or",
                                                                                 "conditions": [{"value": {
                                                                                     "left": {"type": "ref",
                                                                                              "content": ["start_0",
                                                                                                          "enable"]},
                                                                                     "operator": "is_true"}}]}]}},
            {"id": "code_q6CmU", "type": "code",
             "data": {"inputsValues": {"a": {"type": "ref", "content": ["start_0", "query"]}}}}
        ],
        "edges": [
            {"sourceNodeID": "start_0", "targetNodeID": "condition_alRhO"},
            {"sourceNodeID": "condition_alRhO", "targetNodeID": "code_q6CmU", "sourcePortID": "branch_R_7LT"},
            {"sourceNodeID": "condition_alRhO", "targetNodeID": "code_q6CmU", "sourcePortID": "else"},
            # 这里假设 else 也去 code 用于演示
            {"sourceNodeID": "code_q6CmU", "targetNodeID": "end_0"}
        ]
    }

    # 1. 编译图
    compiler = GraphCompiler(flow_data)
    app = compiler.compile()

    # 2. 准备初始输入
    # 模拟用户输入，覆盖 Start 节点的默认值
    inputs = {
        "inputs": {
            "query": "Hello from LangGraph",
            "enable": True  # 设为 True 触发分支
        },
        "node_outputs": {}
    }

    # 3. 执行
    print(">>> Start Workflow Execution")
    try:
        final_state = app.invoke(inputs)
        print("\n>>> Execution Finished Successfully")
        # 打印 End 节点的最终产出
        print("Final Result:", final_state["node_outputs"].get("end_0"))
    except Exception as e:
        print("Execution Error:", e)