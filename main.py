import logging
from pymilvus import MilvusClient
from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
os.environ.get()


DB_NAME = "airene_memory"
OPENAI_KEY = os.getenv("OPENAI_API")

llm = OpenAI()
chat_model = ChatOpenAI(api_key=OPENAI_KEY, model="gpt-3.5-turbo-0125")

client = MilvusClient(DB_NAME)

if not client.has_collection("people"):
    client.create_collection(
        collection_name="people",
        dimension=384
    )
    logging.info("Created collection 'people'")


def search_memory(collection, search_string: str) -> list:
    output = client.search(
        collection_name=collection,

    )
    return output


def add_memory(memories: list):
    ...

