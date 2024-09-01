from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agisample.framework.graph.customersupport.BookCar import search_car_rentals, update_car_rental, book_car_rental, \
    cancel_car_rental
from agisample.framework.graph.customersupport.BookHotel import search_hotels, book_hotel, update_hotel, cancel_hotel
from agisample.framework.graph.customersupport.FetchFlightInfo import fetch_user_flight_information, search_flights, \
    update_ticket_to_new_flight, cancel_ticket
from agisample.framework.graph.customersupport.LookupPolicy import lookup_policy
from agisample.framework.graph.customersupport.SupportTrip import search_trip_recommendations, update_excursion, \
    cancel_excursion, book_excursion


_ = load_dotenv(find_dotenv())


# Haiku is faster and cheaper, but less accurate
llm = ChatOpenAI(model="gpt-4o")
# You could swap LLMs, though you will likely want to update the prompts when
# doing so!
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())


# "Read"-only tools (such as retrievers) don't need a user confirmation to use
part_1_safe_tools = [
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_car_rentals,
    search_hotels,
    search_trip_recommendations,
]

# These tools all change the user's reservations.
# The user has the right to control what decisions are made
part_1_sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]
sensitive_tool_names = {t.name for t in part_1_sensitive_tools}
# Our LLM doesn't have to know which nodes it has to route to. In its 'mind', it's just invoking functions.
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    part_1_safe_tools + part_1_sensitive_tools
)