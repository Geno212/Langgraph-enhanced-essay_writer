from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# Define the schema for the state
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


def create_advanced_essay_graph():
    """
    Function to create and return the main state graph for the advanced essay workflow.
    """
    memory = MemorySaver()

    # Define the state graph structure using AgentState schema
    builder = StateGraph(AgentState)

    # Add nodes to the graph
    builder.add_node("planner", lambda state: {})
    builder.add_node("research_plan", lambda state: {})
    builder.add_node("generate", lambda state: {})
    builder.add_node("reflect", lambda state: {})
    builder.add_node("research_critique", lambda state: {})

    # Define edges and transitions
    builder.set_entry_point("planner")
    builder.add_edge("planner", "research_plan")
    builder.add_edge("research_plan", "generate")
    builder.add_conditional_edges(
        "generate",
        lambda state: END if state["revision_number"] > state["max_revisions"] else "reflect",
        {END: END, "reflect": "reflect"}
    )
    builder.add_edge("reflect", "research_critique")
    builder.add_edge("research_critique", "generate")

    # Compile the graph
    return builder.compile(checkpointer=memory)


def visualize_advanced_essay_graph():
    """
    Function to visualize the advanced essay state graph as an image and save it to a file.
    """
    graph = create_advanced_essay_graph()
    try:
        # Generate the graph visualization
        graph_visual = graph.get_graph(xray=1)  # Enable detailed graph representation
        image_data = graph_visual.draw_mermaid_png()  # Create the graph as a PNG image using Mermaid

        # Save the image to a file
        with open('advanced_essay_graph.png', 'wb') as f:
            f.write(image_data)
        print("Graph visualization saved as advanced_essay_graph.png")
    except Exception as e:
        print(f"An error occurred during graph visualization: {e}")

if __name__ == "__main__":
    visualize_advanced_essay_graph()
