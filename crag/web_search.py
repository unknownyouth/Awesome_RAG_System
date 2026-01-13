
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)

from global_state import GlobalState
from langchain_core.documents import Document

def web_search_node(state: GlobalState):
    """
    Search the web for the most relevant information.
    """
    query = state["rewritten_query"]
    documents = state["reranked_documents"]

    docs = web_search_tool.invoke({"query": query})

    web_results = "\\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    documents.append(web_results)

    return {"final_documents": documents}