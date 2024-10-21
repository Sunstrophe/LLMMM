import logging
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
os.environ.get()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DB_NAME = "airene_memory"
OPENAI_API_KEY = os.getenv("OPENAI_API")

llm = OpenAI()
chat_model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY, model="text-embedding-3-large")

vector_store = Chroma(
    collection_name="memory",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

