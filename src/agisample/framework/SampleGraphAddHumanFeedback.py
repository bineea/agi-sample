from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state):
    print("---Step 1---")
    pass


def human_feedback(state):
    print("---human_feedback---")
    pass


def step_3(state):
    print("---Step 3---")
    pass

def step_4(state):
    print("---Step 4---")
    pass

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_node("step_4", step_4)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", "step_4")
builder.add_edge("step_4", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

thread = {"configurable": {"thread_id": "1"}}
for event in graph.stream({"input": "hello world"}, thread):
    print("第一个循环")
    print(event)

# Get user input
user_input = input("Tell me how you want to update the state: ")

# We now update the state as if we are the human_feedback node
graph.update_state(thread, {"input": user_input}, as_node="human_feedback")

for event in graph.stream(None, thread):
    print("第二个循环")
    print(event)

# 如何设定了stream_mode="values"，则只返回状态值,而不是完整的事件对象，同时因为状态值（即第一个参数）设置为None，所以状态值始终为空，导致event就是空
# for event in graph.stream(None, thread, stream_mode="values"):
#     print("第二个循环")
#     print(event)



