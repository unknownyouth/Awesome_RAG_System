import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from query.query_transformation import rewrite_query_node, multi_query_node, route_after_rewrite
from query.query_routing import query_routing_node
from global_state import GlobalState
from langgraph.graph import StateGraph
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(temperature=0,
                 model_name="deepseek-chat",
                 api_key=os.getenv('DEEPSEEK_API_KEY'),
                 base_url="https://api.deepseek.com", 
                 max_tokens=4000)

def generation_node(state: GlobalState):
    """
    Generation node.
    """
    system_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
    """
    chichat_prompt = PromptTemplate(
        template= state["original_question"] 
    )

    if state["search_needed"]:
        chain = system_prompt | llm
        result = chain.invoke({
            "question": state["rewritten_query"],
            "context": state["documents"]
        })
    else:
        chain = chichat_prompt | llm
        result = chain.invoke({
            "question": state["original_question"],
        })
    
    return {"final_answer": result.content}

def build_graph():
    """
    Build the graph.
    """
    graph = StateGraph(GlobalState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("query_routing", query_routing_node)
    graph.add_node("multi_query", multi_query_node)
    graph.add_node("generation", generation_node)
    # graph.add_edge(START, "rewrite_query")
    # graph.add_edge("rewrite_query", "route_after_rewrite")
    # graph.add_edge("route_after_rewrite", "multi_query")
    # graph.add_edge("multi_query", END)
    graph.set_entry_point("rewrite_query")

    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "query_routing": "query_routing",
            "generation": "generation"
        }
    )
    graph.add_edge("query_routing", "multi_query")
    graph.add_edge("multi_query", END)
    graph.add_edge("generation", END)
    return graph.compile()

whole_graph = build_graph()

def main():
    """
    Main function to run the query transformation pipeline.
    """
    # graph = StateGraph(GlobalState)
    # graph.add_node("rewrite_query", rewrite_query_node)
    # graph.add_node("multi_query", multi_query_node)
    # graph.add_node("generation", generation_node)
    # # graph.add_edge(START, "rewrite_query")
    # # graph.add_edge("rewrite_query", "route_after_rewrite")
    # # graph.add_edge("route_after_rewrite", "multi_query")
    # # graph.add_edge("multi_query", END)
    # graph.set_entry_point("rewrite_query")

    # graph.add_conditional_edges(
    #     "rewrite_query",
    #     route_after_rewrite,
    #     {
    #         "multi_query": "multi_query",
    #         "generation": "generation"
    #     }
    # )
    # graph.add_edge("multi_query", END)
    query_transformation_graph = build_graph()
    result = query_transformation_graph.invoke({"original_question": "Hi, how are you?", "chat_history": []})
    print(result)



if __name__ == "__main__":
    main()