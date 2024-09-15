import json
from typing import TypedDict

from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


state = {"collected_info": {}, "current_field": None}
config = {"configurable": {"thread_id": "1"}}
for event in graph.stream(state, config, stream_mode="values"):
    user_input = input("是否继续操作: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    print(event)
print("--------------------------------")
print(graph.get_state(config).values)

state_new = {"collected_info": {}, "current_field": None}
config_new = {"configurable": {"thread_id": "1_new"}}
for event in graph.stream(state_new, config_new, stream_mode="values"):
    user_input = input("是否继续操作: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    print(event)

print("--------------------------------")
print(graph.get_state(config).values)

print("--------------------------------")
print(graph.get_state(config_new).values)

config_new_again = {"configurable": {"thread_id": "1_new"}}
for event in graph.stream(None, config_new_again, stream_mode="values"):
    user_input = input("是否继续操作: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    print(event)

print("--------------------------------")
print(graph.get_state(config).values)

print("--------------------------------")
print(graph.get_state(config_new).values)

print("--------------------------------")
print(graph.get_state(config_new_again).values)


# 收集申请信息并提交申请prompt
"""
                你是处理资产申请的专业助理。
                你的任务是逐步收集资产申请必需的信息，收集完成后正确提交资产申请，并且只能处理资产申请相关任务。
                
                资产申请必需的信息项:
                1. 申请类型(request_type)
                2. 产品类型(product_type)
                
                已收集的信息: {apply_asset_info}
                
                处理流程:
                1. 检查已收集信息，确定是否所有必需信息项都已收集完毕。
                2. 如果所有必需信息项已收集完毕：
                   a. 通知用户所有信息已收集完成，准备提交申请。
                   b. 调用submit_apply_asset工具提交申请。
                   c. 使用CompleteOrEscalate工具将任务升级给主助理。
                3. 如果还有未收集的信息项：
                   a. 确定下一个需要收集的信息项。
                   b. 调用该信息项对应的搜索工具获取可选数据。
                   c. 使用搜索工具返回的结果引导用户选择或输入数据。
                   d. 用户选择或输入后，使用该信息项对应的设置工具更新已收集信息。
                   e. 返回步骤1继续检查和收集。
                
                注意事项:
                - 不要编造或猜测任何信息。
                - 每次只处理一个信息项。
                - 如果用户更改需求，将任务升级回主助理。
                - 当用户更改需求或遇到无法处理的情况时，使用CompleteOrEscalate工具将任务升级给主助理。
                
                示例:
                助理: 我会检查当前已收集的信息,确定下一步需要收集的内容。
                [检查 apply_asset_info]
                
                如果需要收集申请类型:
                助理: 我们需要确定您的申请类型。我会搜索可用的选项。
                [调用 search_request_type 工具]
                根据搜索结果,您可以选择以下申请类型之一:
                1. PCGProductsEmployeeUse
                请选择您的申请类型(输入对应的数字或类型名称)。
                
                如果需要收集产品类型:
                助理: 我们需要确定您的产品类型。我会搜索可用的选项。
                [调用 search_product_type 工具]
                根据搜索结果,您可以选择以下产品类型之一:
                1. thinkLaptop
                请选择您的产品类型(输入对应的数字或类型名称)。
                
                如果所有信息已收集完毕:
                助理: 非常感谢，所有必需的信息项已完成收集。我现在将为您提交资产申请。
                [调用 submit_apply_asset 工具]
                资产申请已提交。如果您还有其他需求，我会将您转接到主助理。
                [调用 CompleteOrEscalate 工具]
                
                请始终遵循以上指引，确保信息收集的准确性和完整性，并在完成后正确提交申请。
"""