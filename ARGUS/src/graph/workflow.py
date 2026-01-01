from langgraph.graph import StateGraph
from src.graph.state import GraphState
from src.graph.nodes import retrieve_and_grade, should_retry, rewrite_query


def build_graph(vectordb):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", lambda s: retrieve_and_grade(s, vectordb))
    graph.add_node("rewrite", rewrite_query)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        should_retry,
        {
            "rewrite": "rewrite",
            "generate": "__end__",
        }
    )

    graph.add_edge("rewrite", "retrieve")

    return graph.compile()
