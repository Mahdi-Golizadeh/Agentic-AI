from typing import List, TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    query: str
    documents: List[Document]
    relevant_docs: List[Document]
    retry_count: int
