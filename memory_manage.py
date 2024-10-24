from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv
import logging

load_dotenv()


def store_memory(vector_db: Chroma, messages: list[dict]):
    OPENAI_API_KEY = os.getenv("OPENAI_API")
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo-0125")
    system_message = """
    Given the following conversation logs, only extract the parts of the conversation that are relevant for memory. Ignore jokes or casual talk. Focus on important information like tasks, goals, technical details, or instructions.
    "message_id", "message_type" (human or ai), and "content". 

    Conversation:
    {}
    
    Relevant memory pieces (in JSON format):
    """
    prompt = system_message.format(messages)

    res = llm.invoke(prompt)

    try:
        memory_pieces = json.loads(res)

    except json.JSONDecodeError:
        logging.warning("Failed to decode memory_pieces")
        memory_pieces = []

    return memory_pieces
