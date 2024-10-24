import logging
import json
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "airene_memory"
OPENAI_API_KEY = os.getenv("OPENAI_API")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")

# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    # Update message history with response:
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

log_directory = ".logs"

if not os.path.exists(log_directory):
    os.makedirs(log_directory)


def dump_history(messages: list):
    dumpable_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            dumpable_messages.append({
                "type": "human",
                "content": message.content
            })
        elif isinstance(message, AIMessage):
            dumpable_messages.append({
                "type": "ai",
                "content": message.content
            })
        else:
            dumpable_messages.append({
                "type": "other",
                "content": message.content
            })
    history = json.dumps(dumpable_messages, indent=4)
    with open(".logs/history.json", "w") as file:
        file.write(history)


if __name__ == "__main__":
    print("Welcome to my chatbot. Enter below to start talking.")
    try:
        while True:
            query = input("")
            input_messages = [HumanMessage(query)]
            output = app.invoke({"messages": input_messages}, config)
            # output contains all messages in state
            output["messages"][-1].pretty_print()
    finally:
        dump_history(output["messages"])
