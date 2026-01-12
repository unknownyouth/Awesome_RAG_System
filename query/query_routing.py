import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from global_state import GlobalState

from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from langchain_openai import ChatOpenAI
import os
from typing import Literal
class QueryRoutingOutput(BaseModel):
    routing_decision: Literal["vector_store", "graph_database", "relational_database"] = Field(
        description="The single best data source to answer the query."
    )

llm = ChatOpenAI(temperature=0,
                 model_name="deepseek-chat",
                 api_key=os.getenv('DEEPSEEK_API_KEY'),
                 base_url="https://api.deepseek.com", 
                 max_tokens=4000)

def query_routing_node(state: GlobalState):
    """
    Query routing node.
    """
    rewritten_query = state["rewritten_query"]
    
    system_prompt = """You are a routing module in a Retrieval-Augmented Generation (RAG) system.

Your task is to decide which single data source is most appropriate to retrieve information for the user’s query.

There are exactly THREE possible sources:

1. vector_store
   - Unstructured or semi-structured text documents (articles, reports, emails, web pages, notes, PDFs).
   - Best for semantic search, long-form explanations, conceptual questions, and “tell me about X” style queries.
   - Use when the user is asking for narratives, descriptions, definitions, explanations, or general knowledge.

2. graph_database
   - A knowledge graph of entities and relationships between them.
   - Best for questions about connections, paths, multi-hop reasoning, or relational structure, such as:
     * “How is A related to B?”
     * “What paths connect X and Y?”
     * “Who worked with both A and B?”
   - Use when the query explicitly or implicitly focuses on relationships, networks, hierarchies, or multi-step links.

3. relational_database
   - Structured tabular data (rows/columns) with well-defined fields: e.g. tables of students, matches, scores, transactions, log records.
   - Best for questions involving:
     * Exact filtering: “all records where…”
     * Aggregation: counts, sums, averages, minima/maxima
     * Comparisons over numeric or categorical fields
     * Time-series over structured logs
   - Use when the query sounds like it could be answered by SQL over tables (SELECT, WHERE, GROUP BY, ORDER BY).

CRITICAL RULES:
- You MUST strongly prefer relational_database when the query clearly asks for filtering, counting, aggregating, or comparing structured records.
- You MUST strongly prefer graph_database when the query is about relationships, paths, or “who/what connects A and B”.
- ONLY choose vector_store if the question is primarily about free-text content, explanation, or narrative, and does NOT clearly match the graph or relational patterns.
- Do NOT default to vector_store when the query obviously matches graph_database or relational_database patterns.
- ALWAYS choose exactly ONE source.

Here are some examples:

[Example 1]
User query: "List all matches in 2023 where our team scored more than 20 points."
Correct source: relational_database

[Example 2]
User query: "Which players have played for both Team A and Team B?"
Correct source: graph_database

[Example 3]
User query: "Explain how our training strategy changed between 2023 and 2024."
Correct source: vector_store

Now, decide the best source for the following query.

User query:
{rewritten_query}

{format_instructions}
"""
    parser = PydanticOutputParser(pydantic_object=QueryRoutingOutput)
        
    format_instructions = parser.get_format_instructions()
        
        # 4. 构建 Prompt，必须包含 {format_instructions}
    prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["rewritten_query"],
            partial_variables={"format_instructions": format_instructions},
        )
        
        # 5. 组合 Chain
        # 注意：这里不用 with_structured_output，而是用普通的管道
    chain = prompt | llm | parser

    result: QueryRoutingOutput = chain.invoke({
        "rewritten_query": rewritten_query
    })
    
    return {"routing_decision": result.routing_decision}

def route_to_database(state: GlobalState):
    """
    Route to database for retrieval.
    """
    routing_decision = state["routing_decision"]
    if routing_decision == "vector_store":
        return "vector_store"
    elif routing_decision == "graph_database":
        return "graph_database"
    elif routing_decision == "relational_database":
        return "relational_database"

def route_after_rewrite(state: GlobalState):
    """
    根据 rewrite 结果决定下一步去哪
    """
    if state["search_needed"]:
        return "query_routing"
    else:
        return "generation"        # 指向直接回复流程 (跳过检索)
