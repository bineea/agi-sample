from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())

class AskHuman(BaseModel):
    """Ask the human a question"""

    question: str


model = ChatOpenAI(model="gpt-4o")
model = model.bind_tools([AskHuman])

# LLM绑定tool后，LLM会触发tool_calls
print(model.invoke([HumanMessage(content="Use the search tool to ask the user where they are, then look up the weather there")]))