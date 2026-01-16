构建一个**工业级（Production-Grade）的 RAG 系统，与我们在 Colab 里跑的 Demo 有本质区别。工业级系统必须解决高并发、低延迟、数据更新、幻觉控制**以及**可观测性**等问题。

结合你之前关注的 **Hybrid Search**、**Re-ranking** 和 **CRAG**，我为你设计了一套完整的端到端架构。我们将系统分为 **“离线数据流 (Data Pipeline)”** 和 **“在线服务流 (Service Pipeline)”** 两大部分。

---

### 第一部分：离线数据链路 (Offline ETL Pipeline)

**目标**：不仅仅是“存进去”，而是“高质量地存进去”。

### 1. 多模态解析 (Advanced Parsing)

- **痛点**：PDF 里的表格、图片、双栏排版是 RAG 的噩梦。
- **工业级方案**：
    - **工具**：使用 **Unstructured.io** 或 **LlamaParse**（专门针对复杂文档优化）。
    - **策略**：OCR 识别表格转为 Markdown/HTML；图片经过 Multimodal Model (如 GPT-4o-mini) 生成文本描述。

### 2. 智能切分 (Semantic Chunking)

- **痛点**：固定字符数（Fixed-size chunking）会切断语义。
- **工业级方案**：Chroma！
    - **Semantic Chunking**：基于 Embedding 相似度突变点来切分，保证一个 Chunk 说完一件完整的事。
    - **Parent Document Retrieval**（必须上）：
        - **存**：存大块（Parent Chunk，2000 tokens）以保留完整上下文。
        - **搜**：索引小块（Child Chunk，300 tokens）以提高搜索精度。

### 3. 索引构建 (Indexing Strategy)

- **痛点**：单一向量索引无法处理关键词搜索。
- **工业级方案**：
    - **Vector Index**: 使用 **HNSW** 算法（Milvus/Qdrant）。
    - **Inverted Index (BM25)**: 同时建立关键词倒排索引（Elasticsearch/Qdrant 混合支持）。
    - **Metadata**: 提取 `Title`, `Page`, `Date`, `Category` 存入 metadata，用于后续的 **Pre-filtering**。

**第二部分：在线服务链路 (Online Serving Pipeline)**
**目标**：精准理解用户意图，并在 300ms 内找回最正确的文档。
**1. 查询预处理 (Query Understanding & Routing) —— *Agentic 的入口***
• **Guardrails**: 检测用户输入是否包含敏感词或攻击指令 (NeMo Guardrails)。
• **Query Rewrite**: 使用 LLM 将 "它多少钱？" 改写为 "iPhone 15 Pro 的价格是多少？"。
• **Multi Query** (vLLM + Qwen2.5-7B-Instruct)
• **Router (语义路由)**:
    ◦ 如果是“总结全文” $\rightarrow$ 走 Summary Index。
    ◦ 如果是“对比参数” $\rightarrow$ 走 SQL/Tabular 检索。
    ◦ 如果是“普通问答” $\rightarrow$ 走 Vector Store。
**2. 混合检索与重排 (Hybrid Retrieval + Re-ranking) —— *核心***
这是保证 **Recall** 和 **Precision** 的黄金组合。
• **Step 1: 并行召回 (Recall)**
    ◦ 路 A：**Vector Search** (Top 100) $\rightarrow$ 抓语义。
    ◦ 路 B：**BM25 Search** (Top 100) $\rightarrow$ 抓关键词/型号。
    ◦ **Fusion**: 使用 **RRF (Reciprocal Rank Fusion)** 合并得到 Top 100。
• **Step 2: 重排序 (Re-ranking)**
    ◦ 使用 **Cross-Encoder 模型** (如 `bge-reranker-v2-m3` 或 `Cohere Rerank API`)。
    ◦ 对 Top 100 进行精细打分，掐头去尾，只留 **Top 5-10**。
**3. 结果评估与纠错 (Corrective Strategy) —— *CRAG***
• **Lightweight Evaluator**: 引入一个极小的评分模型（或 LLM Prompt）。
• **判断逻辑**：
    ◦ 如果 Top 1 分数 > 0.8 $\rightarrow$ 直接生成。
    ◦ 如果分数 < 0.5 $\rightarrow$ 触发 **Web Search (Tavily/Serper)** 补充外部信息。
**4. 生成与后处理 (Generation)**
• **Prompt Engineering**: 动态注入 System Prompt，要求必须带有 **[Citation]**。
• **Streaming**: 必须支持打字机效果（SSE），降低用户感知的延迟。
**第三部分：MLOps 与可观测性 (Evaluation & Observability)**
**目标**：不能盲跑，必须知道系统每一环节的表现。
**1. 评估体系 (Evaluation)**
不要只凭感觉，要看数据。
• **工具**: **Ragas** 或 **TruLens**。
• **指标**:
    ◦ **Context Recall**: 召回率够吗？
    ◦ **Faithfulness**: 回答有没有幻觉？
    ◦ **Answer Relevancy**: 回答偏题了吗？
**2. 追踪与监控 (Tracing)**
• **工具**: **LangSmith** (LangChain) 或 **Arize Phoenix** (LlamaIndex)。
• **监控**: 记录每一次 Query 的 Trace，查看是哪一步（检索、重排、LLM）耗时最长。
**3. 缓存层 (Caching)**
• **Semantic Cache (Redis/GPTCache)**:
    ◦ 如果用户问“如何重置密码”，两分钟后另一个用户问“密码怎么重置”。
    ◦ 语义缓存发现这两个向量极度相似，直接返回之前的答案，**跳过整个 RAG 流程**，省钱又快。
**技术栈推荐 (Tech Stack Recommendation)**
作为 CS Master，这套技术栈既符合工业界潮流，也适合写进简历：**模块推荐技术栈理由OrchestrationLlamaIndex** (优于 LangChain)在 RAG 数据结构和检索策略上更专业、更深。**Vector DBQdrant** 或 **Milvus**Qdrant 对 Hybrid Search 支持极好，Rust 写的，快。**EmbeddingBGE-M3** (本地) 或 **OpenAI**BGE-M3 支持多语言、长文本和稀疏向量。**Re-rankerBGE-Reranker-v2-m3**目前开源界最强重排模型之一。**LLM InferencevLLM** (部署开源模型)生产环境部署 Llama 3 / Qwen 的吞吐量之王。**EvaluationRagas**工业界标准的 RAG 评分框架。**API FrameworkFastAPI**Python 后端标准。
**总结：下一步你可以做什么？**
建议你不要试图一次性把所有代码写完。按照以下顺序构建：
1. **MVP**: PDF解析 + 简单的 Hybrid Search + LLM。
2. **Add Re-ranking**: 加入重排模型，你会发现准确率大幅提升。
3. **Add Eval**: 接入 Ragas，跑个分，看看瓶颈在哪。
4. **Add Advanced**: 最后再考虑 Query Routing 和 CRAG。

structured output:
对于不支持with_structured_output()
1.使用 PydanticOutputParser (LangChain 标准解法)
LangChain 提供了一个专门的解析器，它会自动生成一段“提示词指令（Format Instructions）”，告诉模型该如何格式化输出。

2.方案二：Few-Shot Prompting (少样本提示)
如果不使用 Function Calling，模型很容易“自由发挥”。最有效的约束手段是给它看例子 (Few-Shot)。
你需要在 Prompt 里明确写出 Input 和 JSON Output 的对应关系。

3.方案三：自动修复机制 (OutputFixingParser)
对于不支持格式化的弱模型，最常见的问题是：
JSON 格式错误（比如少个引号，或者用了中文逗号）。
输出了废话前缀（比如 "Sure! Here is the JSON: ..."）。
LangChain 提供了一个 Auto-Fixing 机制。如果第一次解析失败了，它会把“错误的输出”和“报错信息”一起扔回给 LLM，让 LLM 自己修。

4.方案四：受限解码 (Constrained Decoding) —— 仅限本地部署
如果你使用的是 vLLM 本地部署（你之前提到了 Qwen 2.5），这是最强、最完美的方案。
你可以直接在推理引擎层面“禁止”模型生成任何不符合 JSON 语法的 Token。这不需要模型“聪明”，而是从数学概率上锁死了它只能输出 JSON。

Query Router training Roadmap：
Dataset Building：
Pubic Dataset：
Relational Database： Spider WikiSQL
Graph Database： MetaQA WebQuestionsSP ComplexWebQuestions
Vector Store： Natural Questions(google ai) HotpotQA(Explaination Style) TriviaQA

Data Distribution:
Public:65%
Synthetic:35% (40% easy, 60% hard)

Data Generation Prompts:

You are generating training data for a query routing classifier in a RAG system.

Generate 20 user queries that should be answered using a VECTOR STORE.

These queries should:
- Ask for explanations, summaries, definitions, or conceptual understanding
- Be answerable using unstructured or semi-structured text documents
- NOT require counting, aggregation, filtering over tables
- NOT require explicit reasoning over entity relationships or paths

Output a JSON list.
Each item must follow this schema:

{
  "query": "...",
  "label": "vector_store",
  "difficulty": "easy",
  "rationale": "..."
}

You are generating HARD training examples for a query routing classifier.

Generate 20 user queries that should STILL be answered using a VECTOR STORE,
but which could easily be confused with graph or relational queries.

These queries should:
- Mention entities, years, or comparisons
- Possibly include words like "compare", "difference", or "trend"
- BUT ultimately require narrative explanation or textual synthesis,
  not structured querying or relationship traversal

Output a JSON list using this schema:

{
  "query": "...",
  "label": "vector_store",
  "difficulty": "hard",
  "rationale": "..."
}

You are generating training data for a query routing classifier.

Generate 20 user queries that should be answered using a RELATIONAL DATABASE.

These queries should:
- Clearly involve structured records stored in tables
- Require filtering, aggregation, sorting, or grouping
- Sound like they could be translated into SQL

Include queries involving:
- counts, averages, sums
- conditions like WHERE, BETWEEN, >, <, =
- GROUP BY or ORDER BY semantics

Output a JSON list with this schema:

{
  "query": "...",
  "label": "relational_database",
  "difficulty": "easy",
  "rationale": "..."
}

Generate 20 HARD user queries that should be answered using a RELATIONAL DATABASE.

These queries should:
- Be phrased in natural language, not explicit SQL
- Hide their structured nature behind conversational wording
- Still fundamentally require querying structured tables

They may:
- Ask for comparisons across years
- Ask for rankings or top-k items
- Combine multiple conditions

Output a JSON list with this schema:

{
  "query": "...",
  "label": "relational_database",
  "difficulty": "hard",
  "rationale": "..."
}

Generate 20 user queries that should be answered using a GRAPH DATABASE.

These queries should:
- Ask about relationships between entities
- Require traversing one or more edges in a knowledge graph
- Involve phrases like "connected to", "worked with", "related to", "shared with"

Do NOT include:
- Aggregation over numeric fields
- Pure explanation or summarization

Output a JSON list using this schema:

{
  "query": "...",
  "label": "graph_database",
  "difficulty": "easy",
  "rationale": "..."
}

Generate 20 HARD user queries that should be answered using a GRAPH DATABASE.

These queries should:
- Be phrased ambiguously or conversationally
- Involve multiple entities and implicit relationships
- Require reasoning over connections or shared attributes
- Be easily confused with vector-store explanation queries

Avoid:
- Queries that are purely descriptive
- Queries that only require counting or aggregation

Output a JSON list with this schema:

{
  "query": "...",
  "label": "graph_database",
  "difficulty": "hard",
  "rationale": "..."
}
TODO:
Documents refinement
Query Router
Building knowledge sources