from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# Set the OpenAI API key environment variable
os.environ["DEEPSEEK_API_KEY"] = os.getenv('DEEPSEEK_API_KEY')

class QueryTransformationState(TypedDict):

    original_question: str            # user's original question
    chat_history: List[BaseMessage]   # conversation history (LangChain Message object list)
    
    # output fields (filled by nodes)
    rewritten_query: str              # rewritten standard query
    multi_queries: List[str]          # multi-query based on user input and history
    search_needed: bool               # whether to execute search (for routing)
    step_log: List[str]               # (optional) engineering debug log list

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

# --- 3. 定义 Rewrite 节点 ---
def rewrite_query_node(state: QueryTransformationState):
    """
    节点功能: 接收原始问题 + 历史，输出重写后的问题 + 意图
    """
    original_q = state["original_question"]
    history = state.get("chat_history", [])

    # A. 准备 Prompt
    # 技巧: System Prompt 必须强调'指代消解'
    system_prompt = """You are a Query Rewrite Engine for a RAG system.
    1. Resolve pronouns (it, he, that) in the user's latest question using Chat History.
    2. If the user input is just 'hi', 'thanks', or 'bye', set is_chit_chat=True.
    3. Output the standalone query in the same language as the user.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


    # structured_llm = llm.with_structured_output(RewriteOutput)
    # chain = prompt | structured_llm

    
        
        # 2. 初始化解析器
    parser = PydanticOutputParser(pydantic_object=RewriteOutput)
        
        # 3. 获取格式指令 (关键步骤)
        # 这会自动生成一段很长的 String，教模型怎么写 JSON
    format_instructions = parser.get_format_instructions()
        
        # 4. 构建 Prompt，必须包含 {format_instructions}
    prompt = PromptTemplate(
            template="""你是一个查询重写助手。将用户的查询改写为更具体、详细的问题。
            
{format_instructions}

用户输入: {question}

请按照上述 JSON 格式输出结果。""",
            input_variables=["question"],
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
            "question": original_q
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

def route_after_rewrite(state: QueryTransformationState):
    """
    根据 rewrite 结果决定下一步去哪
    """
    if state["search_needed"]:
        return "continue_search" # 指向 RAG 检索流程
    else:
        return "end_chat"        # 指向直接回复流程 (跳过检索)