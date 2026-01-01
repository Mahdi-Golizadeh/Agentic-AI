"""
LangGraph nodes for agentic retrieval with relevance grading
"""

from src.agents.relevance_grader import RelevanceGrader
from src.agents.query_rewriter import QueryRewriter
from src.agents.answer_generator import AnswerGenerator

answer_generator = AnswerGenerator()

def generate_answer(state):
    if len(state["relevant_docs"]) == 0:
        return {
            "answer": "The provided documents do not contain enough information to answer the question."
        }

    answer = answer_generator.generate(
        state["query"],
        state["relevant_docs"]
    )

    return {
        "answer": answer
    }

# -------------------------------
# Global configuration
# -------------------------------

MIN_RELEVANT = 2
MAX_RETRIES = 2
TOP_K = 6

# -------------------------------
# Instantiate agents ONCE
# -------------------------------

# These load the LLM a single time (singleton pattern)
grader = RelevanceGrader()
rewriter = QueryRewriter()

# -------------------------------
# Node: Retrieve + Grade
# -------------------------------

def retrieve_and_grade(state, vectordb):
    """
    Retrieve documents from vector DB and grade their relevance.
    """
    query = state["query"]

    retrieved_docs = vectordb.similarity_search(query, k=TOP_K)
    relevant_docs = []

    for doc in retrieved_docs:
        result = grader.grade(query, doc)

        if result["relevant"] == "yes":
            doc.metadata["relevance_reason"] = result["reason"]
            relevant_docs.append(doc)

    return {
        "query": query,
        "documents": retrieved_docs,
        "relevant_docs": relevant_docs,
        "retry_count": state["retry_count"],
    }

# -------------------------------
# Node: Decision Logic
# -------------------------------

def should_retry(state):
    """
    Decide whether to retry retrieval by rewriting the query.
    """
    # Stop if retry budget exhausted
    if state["retry_count"] >= MAX_RETRIES:
        return "generate"

    # Retry if not enough relevant evidence
    if len(state["relevant_docs"]) < MIN_RELEVANT:
        return "rewrite"

    # Retry if relevance reasoning suggests partial coverage
    for doc in state["relevant_docs"]:
        reason = doc.metadata.get("relevance_reason", "").lower()
        if any(keyword in reason for keyword in ["partial", "limited", "unclear"]):
            return "rewrite"

    return "generate"

# -------------------------------
# Node: Rewrite Query
# -------------------------------

def rewrite_query(state):
    """
    Rewrite the user query to improve retrieval quality.
    """
    new_query = rewriter.rewrite(state["query"])

    return {
        "query": new_query,
        "retry_count": state["retry_count"] + 1,
    }
