import functools
import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.agents import create_agent as create_langchain_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import Field, create_model

from agisample.langgraph.hierarchical_teams.web_tool import create_tavily_tool, scrape_webpages


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
):
    """创建 LangChain v1 agent，并添加团队协作提示。"""
    system_prompt += (
        "\nWork autonomously according to your specialty, using the tools available to you."
        " Do not ask for clarification."
        " Your other team members (and other teams) will collaborate with you with their own specialties."
        " You are chosen for a reason! You are one of the following team members: {team_members}."
    )
    return create_langchain_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


def agent_node(state, agent, name):
    result = agent.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}


def create_team_supervisor(llm: ChatOpenAI, system_prompt: str, members: list[str]):
    """创建基于结构化输出的团队路由器。"""
    options = ["FINISH"] + members
    route_response = create_model(
        "RouteResponse",
        next=(Literal[tuple(options)], Field(description="下一个要执行的节点")),
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return prompt | llm.with_structured_output(route_response)


class ResearchTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str


def build_research_team(model: str = "gpt-4-1106-preview"):
    llm = ChatOpenAI(model=model)

    search_agent = create_agent(
        llm,
        [create_tavily_tool()],
        "You are a research assistant who can search for up-to-date info using the tavily search engine.",
    )
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")

    research_agent = create_agent(
        llm,
        [scrape_webpages],
        "You are a research assistant who can scrape specified urls for more detailed information using the scrape_webpages function.",
    )
    research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")

    supervisor_agent = create_team_supervisor(
        llm,
        "You are a supervisor tasked with managing a conversation between the"
        " following workers:  Search, WebScraper. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.",
        ["Search", "WebScraper"],
    )
    return search_node, research_node, supervisor_agent
