from langgraph.graph import StateGraph, END
from typing import TypedDict
from agent import router, sales_agent, needs_agent
from product_agent import product_agent

class AgentState(TypedDict):
    input: str
    output: str
    decision: str

def create_graph():
    # Initialize the graph
    workflow = StateGraph(AgentState)

    # Add nodes for each agent
    workflow.add_node("router", router)
    workflow.add_node("needs_agent", needs_agent)
    workflow.add_node("sales_agent", sales_agent)
    workflow.add_node("product_agent", product_agent)
    workflow.add_node("recommendation_agent", lambda x: {"output": recommendation_agent(x)["output"]})

    # Add conditional edges from router to agents
    workflow.add_conditional_edges(
        "router",
        lambda x: x["decision"],
        {
            "sales_agent": "sales_agent",
            "needs_agent": "needs_agent",
            "product_agent": "product_agent",
            "recommendation_agent": "recommendation_agent"
        }
    )

    # Set entry point and end points
    workflow.set_entry_point("router")
    workflow.add_edge("sales_agent", END)
    workflow.add_edge("product_agent", END)
    workflow.add_edge("needs_agent", END)
    workflow.add_edge("recommendation_agent", END)

    # Compile and return the graph
    return workflow.compile()

# Export the create_graph function
__all__ = ['create_graph']