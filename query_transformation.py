from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langsmith import traceable
from global_state import GlobalState
load_dotenv()
# Set the OpenAI API key environment variable
os.environ["DEEPSEEK_API_KEY"] = os.getenv('DEEPSEEK_API_KEY')

# os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
# os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
# os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# class QueryTransformationState(TypedDict):

#     original_question: str            # user's original question
#     chat_history: List[BaseMessage]   # conversation history (LangChain Message object list)
    
#     # output fields (filled by nodes)
#     rewritten_query: str              # rewritten standard query
#     multi_queries: List[str]          # multi-query based on user input and history
#     search_needed: bool               # whether to execute search (for routing)
#     step_log: List[str]               # (optional) engineering debug log list

# define the structured output schema for the LLM (Pydantic)
class RewriteOutput(BaseModel):
    """The strongly constrained format of the LLM output"""
    rewritten_query: str = Field(
        description="The standalone, refined search query based on user input and history."
    )
    is_chit_chat: bool = Field(
        description="True if the user input is just greeting/thanks and needs no search. False if it's a question."
    )

# We generate multiple queries after rewriting the query, so we do not need to return is_chit_chat here.
class MultiQueryOutput(BaseModel):
    """The strongly constrained format of the LLM output"""
    multi_queries: List[str] = Field(
        description="The list of multi-query based on user input and history."
    )

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 初始化 LLM (假设你已经配好了 .env) ---
# 建议: Rewrite 任务使用 temperature=0 以保证稳定
llm = ChatOpenAI(temperature=0,
                 model_name="deepseek-chat",
                 api_key=os.getenv('DEEPSEEK_API_KEY'),
                 base_url="https://api.deepseek.com", 
                 max_tokens=4000)


def rewrite_query_node(state: GlobalState):
    """
    节点功能: 接收原始问题 + 历史，输出重写后的问题 + 意图
    """
    original_q = state["original_question"]
    history = state.get("chat_history", [])

    # A. 准备 Prompt
    # 技巧: System Prompt 必须强调'指代消解'
    system_prompt = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {original_query}

        {format_instructions}

        Rewritten query:"""
    


    # structured_llm = llm.with_structured_output(RewriteOutput)
    # chain = prompt | structured_llm

    
        
        # 2. 初始化解析器
    parser = PydanticOutputParser(pydantic_object=RewriteOutput)
        
        # 3. 获取格式指令 (关键步骤)
        # 这会自动生成一段很长的 String，教模型怎么写 JSON
    format_instructions = parser.get_format_instructions()
        
        # 4. 构建 Prompt，必须包含 {format_instructions}
    prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["original_query"],
            partial_variables={"format_instructions": format_instructions},
        )
        
        # 5. 组合 Chain
        # 注意：这里不用 with_structured_output，而是用普通的管道
    chain = prompt | llm | parser

    # C. 执行调用
    try:
        # 只取最近 k 轮历史以节省 Token (工程优化)
        recent_history = history[-6:] if history else []
        
        result: RewriteOutput = chain.invoke({
            "history": recent_history,
            "original_query": original_q
        })
        
        # D. 返回状态更新
        # 注意: 这里只返回需要更新的字段，LangGraph 会自动合并到 State
        return {
            "rewritten_query": result.rewritten_query,
            "search_needed": not result.is_chit_chat,
            "step_log": [f"Rewrite: {original_q} -> {result.rewritten_query}"]
        }
        
    except Exception as e: 
        print(f"Rewrite Failed: {e}")
        return {
            "rewritten_query": original_q,
            "search_needed": True,
            "step_log": [f"Rewrite Error: {str(e)}"]
        }


def multi_query_node(state: GlobalState):
    """
    节点功能: 接收重写后的查询，输出多个查询
    """
    rewritten_query = state["rewritten_query"]
    system_prompt = """You are an AI language model assistant. Your task is to generate three different versions of the given user question 
    to retrieve relevant documents from a vector database.
    By generating multiple versions of the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative questions separated by newlines.
    Original query: {rewritten_query}

    {format_instructions}

    Multi-queries:"""

    parser = PydanticOutputParser(pydantic_object=MultiQueryOutput)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(input_variables=["rewritten_query"],
                            template=system_prompt,
                            partial_variables={"format_instructions": format_instructions})
    chain = prompt | llm | parser
    result: MultiQueryOutput = chain.invoke({"rewritten_query": rewritten_query})
    return {
        "multi_queries": result.multi_queries}

def route_after_rewrite(state: GlobalState):
    """
    根据 rewrite 结果决定下一步去哪
    """
    if state["search_needed"]:
        return "multi_query"# 指向 RAG 检索流程
    else:
        return "end_chat"        # 指向直接回复流程 (跳过检索)

def build_query_transformation_graph():
    """
    构建查询转换的 LangGraph
    
    Returns:
        CompiledStateGraph: 编译后的 graph 对象
    """
    graph = StateGraph(GlobalState)
    
    # 添加节点
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("multi_query", multi_query_node)
    
    # 设置入口点
    graph.set_entry_point("rewrite_query")
    
    # 添加条件边：从 rewrite_query 根据路由函数决定下一步
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "multi_query": "multi_query",
            "end_chat": END
        }
    )
    
    # multi_query 节点后直接结束
    graph.add_edge("multi_query", END)
    
    # 编译并返回
    return graph.compile()

# For studio to import
query_transformation_graph = build_query_transformation_graph()

def main():
    """
    Main function to run the query transformation pipeline.
    """
    graph = StateGraph(GlobalState)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("multi_query", multi_query_node)

    # graph.add_edge(START, "rewrite_query")
    # graph.add_edge("rewrite_query", "route_after_rewrite")
    # graph.add_edge("route_after_rewrite", "multi_query")
    # graph.add_edge("multi_query", END)
    graph.set_entry_point("rewrite_query")

    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "multi_query": "multi_query",
            "end_chat": END
        }
    )
    graph.add_edge("multi_query", END)
    query_transformation_graph = graph.compile()
    result = query_transformation_graph.invoke({"original_question": "Hi, how are you?", "chat_history": []})
    print(result)



if __name__ == "__main__":
    main()