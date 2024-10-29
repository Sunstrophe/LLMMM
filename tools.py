from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from langchain_chroma import Chroma
from typing import Type, Optional
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForToolRun


class RememberInput(BaseModel):
    query: str = Field(
        description="The query for what we search the answer to.")


class RememberTool(BaseTool):
    name: str = "Remember"
    description: str = "Used when we want to think and remember something that we might have known before."
    args_schema: Type[BaseModel] = RememberInput
    return_direct: bool = True

    _vector_db: Chroma = PrivateAttr()

    def __init__(self, vector_db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._vector_db = vector_db

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> list[Document]:
        res = self.vector_db.similarity_search(query=query)
        return res
