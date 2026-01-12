# test_retrieval_hf.py

from typing import TypedDict, List
from functools import partial

import chromadb
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from global_state import GlobalState


# ========= 2. 构造一点测试语料，写入 Chroma =========

def build_test_vector_store():
    """
    构造一个小型测试语料，并写入 Chroma。
    每条语料是一个 LangChain Document。
    """
    docs: List[Document] = [
        Document(
            page_content="Volleyball is a team sport that emphasizes cooperation, communication, and strategy.",
            metadata={"id": "doc_volleyball_1", "topic": "sports"},
        ),
        Document(
            page_content="In 2023 and 2024, our university volleyball team won back-to-back provincial championships.",
            metadata={"id": "doc_volleyball_2", "topic": "sports"},
        ),
        Document(
            page_content="Large language models can be fine-tuned using parameter-efficient techniques such as LoRA and QLoRA.",
            metadata={"id": "doc_llm_1", "topic": "AI"},
        ),
        Document(
            page_content="Economics is the study of how individuals and societies allocate scarce resources.",
            metadata={"id": "doc_econ_1", "topic": "economics"},
        ),
    ]
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},   # 如果有 GPU：{"device": "cuda"}
    )

    # 使用单独的测试目录，避免和你正式的 chroma_db 混淆
    persistent_client = chromadb.PersistentClient(path="./chroma_db_test_hf")

    # 如果 collection 已经存在，可以先删掉重建，确保干净
    try:
        persistent_client.delete_collection("hybrid_collection_test_hf")
    except Exception:
        pass

    vector_store = Chroma(
        client=persistent_client,
        collection_name="hybrid_collection_test_hf",
        embedding_function=embedding_function,
    )

    # 将测试文档写入 Chroma
    vector_store.add_documents(docs)

    return vector_store


# ========= 3. 只用 dense retrieval 的节点 =========

def vector_store_retrieval_node(state: GlobalState, vector_store: Chroma):
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


# ========= 4. 构建一个最小 LangGraph，用于测试 =========

def build_app(vector_store: Chroma):
    workflow = StateGraph(GlobalState)

    node_with_dependency = partial(vector_store_retrieval_node, vector_store=vector_store)
    workflow.add_node("vector_store_retrieval", node_with_dependency)

    workflow.add_edge(START, "vector_store_retrieval")
    workflow.add_edge("vector_store_retrieval", END)

    app = workflow.compile()
    return app


# ========= 5. 主函数：构建 DB + 跑一条测试 query =========

def main():
    # 1) 构建测试向量库（使用 HF embeddings）
    vector_store = build_test_vector_store()

    # 2) 构建 graph app
    app = build_app(vector_store)

    # 3) 准备一个测试的 GlobalState 初始值
    init_state: GlobalState = {
        "original_question": "What achievements did the volleyball team have in 2023 and 2024?",
        "multi_queries": [
            "volleyball team championships in 2023 and 2024",
            "university volleyball provincial titles 2023 2024",
        ],
        "documents": [],
    }

    # 4) 运行 graph
    final_state = app.invoke(init_state)

    # 5) 打印检索结果
    print("=== Retrieved Documents ===")
    for i, doc in enumerate(final_state["documents"], start=1):
        print(f"[{i}] content: {doc.page_content}")
        print(f"    metadata: {doc.metadata}")
        print()

if __name__ == "__main__":
    main()
