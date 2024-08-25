import json
from typing import TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})


def generate_question(state):
    # 使用 LLM 根据当前未填写的字段生成下一个问题
    question_prompt = f"""
    你需要引导用户完成信息收集。需要收集的信息：
    1. 用户姓名(name)
    2. 用户手机号(phone)
    3. 用户邮箱(email)
    4. 用户年龄(age)
    5. 用户性别(gender)
    6. 用户职业(job)
    7. 用户月收入(income)
    8. 用户家庭状况(family)
    9. 用户是否有车(car)
    10. 用户是否有房(house)
    目前已经收集的信息为：{state['collected_info']}，只生成一个缺失信息对应的问题。
    以JSON格式输出。
    1.question字段为string类型，表示缺失信息对应的问题内容；
    2.field字段为string类型，表示缺失信息对应的字段；
    例如：
    {{\"question\":\"请问您是否有车？\",\"field\":\"car\"}}
    """
    result = llm.invoke(question_prompt)
    result_data = json.loads(result.content)
    state['current_field'] = result_data['field']
    print(result_data['question'])
    return state


def process_input(state):
   state['collected_info'][state['current_field']] = input("User: ")
   return state


# def validate(state):
#    # 使用 LLM 验证用户输入的信息是否合适
#    prompt = f"请验证以下信息是否合理：{state['collected_info']}，需要校验数据格式，以及数据是否符合语义，只使用Y与N作为最终结论，不要任何分析或描述内容"
#    validation_result = llm.invoke(prompt)
#    print(validation_result)
#    if "Y" in validation_result:
#        return state
#    return state

def check_completion(state):
   if len(state['collected_info']) == 10:
       return "Y"
   return "N"



class State(TypedDict):
    collected_info: {}
    current_field: None

workflow = StateGraph(State)

workflow.add_node("generate_question", generate_question)
workflow.add_node("process_input", process_input)

workflow.set_entry_point("generate_question")
workflow.add_edge("generate_question", "process_input")
workflow.add_conditional_edges("process_input",
                               check_completion,
                               {
                                   "Y": END,
                                   "N": "generate_question"
                               })


graph = workflow.compile()


state = {"collected_info": {}, "current_field": None}
config = {"configurable": {"thread_id": "1"}}
for event in graph.stream(state, config, stream_mode="values"):
    print(event)