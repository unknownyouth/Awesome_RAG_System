from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
import os
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()
from langchain_core.documents import Document

class GlobalState(TypedDict):

    original_question: str            # user's original question
    chat_history: List[BaseMessage]   # conversation history (LangChain Message object list)
    
    # query transformation state
    rewritten_query: str              # rewritten standard query
    multi_queries: List[str]          # multi-query based on user input and history
    search_needed: bool               # whether to execute retrieval (for routing)
    step_log: List[str]               # (optional) engineering debug log list

    # query routing state
    routing_decision: str        # routing decision

    # vector store retrieval state
    documents: List[Document]    # retrieved documents

    # re-ranking state
    reranked_documents: List[Document]    # re-ranked documents

    # retrieval evaluation result
    retrieval_evaluation: List[str]    # retrieval evaluation result
    web_search_needed: bool           # whether to execute web search (for routing)
    final_documents: List[Document]    # final documents for answer generation

    final_answer: str    # answer
