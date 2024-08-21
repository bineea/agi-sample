from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o")

def initialize():
    return {"required_info": ["name", "email", "age"], "collected_info": {}, "current_field": None}

def generate_question(state):
   # 使用 LLM 根据当前未填写的字段生成下一个问题
   question_prompt = f"当前已收集的信息：{state['collected_info']}，需要收集的信息：{state['required_info']}。请生成下一个问题来收集缺失的信息。"
   question = llm.invoke(question_prompt)
   field_prompt = f"收集需要信息的问题：{question}。分析这个问题是针对需要收集的信息：{state['required_info']}中的哪个值？"
   state['current_field'] = llm.invoke(field_prompt)
   return question, state


def process_input(user_input, state):
   state['collected_info'][state['current_field']] = user_input
   return state

def validate(state):
   # 使用 LLM 验证用户输入的信息是否合适
   prompt = f"请验证以下信息是否合理：{state['collected_info']}，需要校验数据格式，以及数据是否符合语义，只使用合理与不合理作为最终结论，不要任何分析或描述内容"
   validation_result = llm.invoke(prompt)
   if "不合理" in validation_result:
       return "重新询问", state
   return "继续", state

def check_completion(state):
   if len(state['collected_info']) == len(state['required_info']):
       return "完成", state
   return "继续", state

from langgraph.graph import StateGraph

workflow = StateGraph()


workflow.add_node("initialize", initialize)
workflow.add_node("generate_question", generate_question)
workflow.add_node("process_input", process_input)
workflow.add_node("validate", validate)
workflow.add_node("check_completion", check_completion)

workflow.set_entry_point("initialize")

workflow.add_edge("initialize", "generate_question")


workflow.add_edge("generate_question", "process_input")
workflow.add_edge("process_input", "validate")



workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("validate", "check_completion")
workflow.add_edge("check_completion", "generate_question")


state = {}
for output in workflow.run(state):
    if isinstance(output, str):
        # 向用户显示问题
        user_input = input(output)
        state = process_input(user_input, state)
    elif output == "完成":
        print("表单填写完成！")
        break