from src.agents.relevance_grader import RelevanceGrader
from src.rag.baseline import BaselineRAG


class GradedRAG(BaselineRAG):
    def __init__(self, vectordb, k=6):
        super().__init__(vectordb, k)
        self.grader = RelevanceGrader()

    def retrieve(self, query):
        docs = self.vectordb.similarity_search(query, k=self.k)

        relevant_docs = []
        for doc in docs:
            grade = self.grader.grade(query, doc)
            if grade["relevant"] == "yes":
                doc.metadata["relevance_reason"] = grade["reason"]
                relevant_docs.append(doc)

        return relevant_docs
