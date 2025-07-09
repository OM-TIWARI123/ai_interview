from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]



llm=init_chat_model("google_genai:gemini-2.0-flash")

def chatbot(state:State):
    system_prompt=SystemMessage(content="""you are a voice agent name silica whose work is to solve the loneliness of people. by taking to them and understanding them, ask them questions about themseleves console them and act as if you are their friend to listen them.""")
    message=llm.invoke([system_prompt]+state["messages"])

    return {"messages":[message]}

graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")

graph=graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
