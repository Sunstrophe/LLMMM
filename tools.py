from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from langchain_chroma import Chroma
from typing import Type
from langchain_core.documents import Document


class RemeberInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed="True")
    vector_db: Chroma = Field(description="Our database to search in.")
    query: str = Field(
        description="Our query for what we search the answer to.")
    

class RemeberTool(BaseTool):
    name: str = "Remeber"
    description: str = "Used when we want to think and remeber something that we might have known before."
    args_schema: Type[BaseModel] = RemeberInput
    return_direct: list[Document]

    def _run(self, vector_db: Chroma, query: str) -> list[Document]:
        res = vector_db.similarity_search(query=query)
        return res

