import logging
from pymilvus import MilvusClient

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DB_NAME = "airene_memory"


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