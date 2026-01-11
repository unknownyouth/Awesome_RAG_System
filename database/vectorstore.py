from chromadb import Search, K, Knn, Rrf
from langchain_core.documents import Document
from typing import TypedDict, List, Any
from global_state import GlobalState


def vector_store_retrieval_node(state: GlobalState):
    """
    LangGraph 节点：执行混合检索 (Dense + Sparse + RRF)
    """
    print("---RETRIEVE---")
    multi_queries = state["multi_queries"]

    # ---------------------------------------------------------
    # 1. 定义 RRF 混合排序逻辑 (复用之前的逻辑)
    # ---------------------------------------------------------
    hybrid_rank = Rrf(
        ranks=[
            # 策略 A: Dense Retrieval (语义搜索)
            Knn(query=query, return_rank=True, limit=50), 
            
            # 策略 B: Sparse Retrieval (关键词搜索)
            # 假设你的 Chroma 集合中存储稀疏向量的字段名为 "sparse_embedding"
            Knn(query=query, key="sparse_embedding", limit=50)
        ],
        weights=[1.0, 1.0] # 权重可调
    )

    # ---------------------------------------------------------
    # 2. 构建搜索配置
    # ---------------------------------------------------------
    search_config = (
        Search()
        .rank(hybrid_rank)
        .limit(5)  # 最终返回给 LLM 的文档数量
        .select(K.DOCUMENT, K.SCORE, "source") # 选择返回字段，根据你的 metadata 调整
    )

    # ---------------------------------------------------------
    # 3. 执行搜索 (假设 vector_store 是外部初始化的 Chroma 对象)
    # ---------------------------------------------------------
    # 注意：在实际应用中，vector_store 应该在外部初始化好
    # 如果 vector_store 是全局变量，直接调用即可
    # 如果不是，建议使用 partial 传入或将此函数放在类方法中
    results = vector_store.hybrid_search(search_config)
    
    # ---------------------------------------------------------
    # 4. 格式化结果以存入 State
    # ---------------------------------------------------------
    # Chroma 返回的结果通常是一个列表，我们需要将其转换为 LangChain Document 对象
    # 或者简单的字符串列表，取决于你后续的 Node 怎么使用
    retrieved_docs = []
    
    # 解析 Chroma 的返回结果 (具体结构取决于 Chroma 版本，这里是通用处理)
    for res in results:
        # 假设 res 包含 document 和 metadata
        content = res.get("document", "")
        # 如果有 metadata，也可以提取
        meta = {"score": res.get("score"), "source": res.get("source", "unknown")}
        
        doc = Document(page_content=content, metadata=meta)
        retrieved_docs.append(doc)

    # 返回更新后的 State (只返回变化的字段)
    return {"documents": retrieved_docs}
# 这里包含两个 Knn 搜索：一个是默认的 Dense (语义)，一个是指定 key 的 Sparse (关键词)
hybrid_rank = Rrf(
    ranks=[
        # 策略 A: Dense Retrieval (稠密向量/语义搜索)
        # return_rank=True 是 RRF 必须的参数，内部 limit 建议设置得比最终 limit 大
        Knn(query="你的搜索关键词", return_rank=True, limit=100), 
        
        # 策略 B: Sparse Retrieval (稀疏向量/关键词匹配)
        # key="sparse_embedding" 指定了去查找稀疏向量字段
        Knn(query="你的搜索关键词", key="sparse_embedding", limit=100)
    ],
    # 权重: 这里设为 [1.0, 1.0] 表示两者同等重要，你可以根据需求调整
    weights=[1.0, 1.0] 
)

# 2. 构建搜索对象 (去掉了 .where 过滤)
search_config = (
    Search()
    .rank(hybrid_rank)              # 应用上面的 RRF 排序
    .limit(10)                      # 最终只取前 10 个结果
    .select(K.DOCUMENT, K.SCORE)    # 只返回文档内容和分数
)

# 3. 执行搜索
results = vector_store.hybrid_search(search_config)