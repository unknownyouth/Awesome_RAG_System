from global_state import GlobalState
from typing import List
from langchain_core.documents import Document



def graph_database_retrieval_node(state: GlobalState, graph_database: GraphDatabase):
    """
    LangGraph 节点：只使用 Chroma 做 dense (语义) 检索。
    - 从 state["multi_queries"] 里取 query
    - 使用 similarity_search
    - 结果写回 state["documents"]
    """
    print("--- DENSE RETRIEVE NODE (HF) ---")

    multi_queries = state.get("multi_queries") or []
    if not multi_queries:
        query = state["original_question"]
        multi_queries = [query]

    all_docs: List[Document] = []

    per_query_k = 3  # 每个子 query 取多少条
    for q in multi_queries:
        print(f"  Query: {q}")
        docs = vector_store.similarity_search(q, k=per_query_k)
        for d in docs:
            md = dict(d.metadata) if d.metadata else {}
            md["source_query"] = q
            d.metadata = md
        all_docs.extend(docs)

    # 简单去重 + 控制总数
    seen = set()
    unique_docs: List[Document] = []
    top_k_global = 5

    for d in all_docs:
        key = d.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
        if len(unique_docs) >= top_k_global:
            break

    print(f"  Retrieved {len(unique_docs)} unique docs.\n")

    return {"documents": unique_docs}