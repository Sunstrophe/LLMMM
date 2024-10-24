import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "airene_memory"
OPENAI_API_KEY = os.getenv("OPENAI_API")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# llm = OpenAI()
chat_model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY, model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

system_message = SystemMessage(content="You are a helpful AI assistant.")


def call_model(state: MessagesState):
    res = chat_model.invoke(state["messages"])
    return {"messages": res}


def conversation_history(message: str):
    ...


def embed_queries(queries: list[dict]):
    for query in queries:
        message = query["message"]
        vector_store.add_texts(texts=message, metadatas=query["time"])
        logging.debug(f"Added {query} to the database")


def prompt_ai(message: str) -> AIMessage:
    prompt = [
        system_message,
        HumanMessage(content=message)
    ]
    res = chat_model.invoke(prompt)
    return res


if __name__ == "__main__":
    print("Input commands below: ")
    while True:
        message = input("")
        res = prompt_ai(message)
        print("---")
        print(res)
        print("---")
        print(type(res))


"""
---
content='The Earth is round due to its gravitational forces pulling towards its center, which causes it to form a spherical shape. 
This shape is known as an oblate spheroid, slightly flattened at the poles and bulging at the equator.' 
additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 48, 'prompt_tokens': 24, 'total_tokens': 72, 
'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}},
 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None} id='run-73dd1bdc-51ef-42f1-9b61-6f0e9e2886e3-0' 
usage_metadata={'input_tokens': 24, 'output_tokens': 48, 'total_tokens': 72, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
---
<class 'langchain_core.messages.ai.AIMessage'>
"""
