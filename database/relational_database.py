from global_state import GlobalState
from langchain_core.documents import Document
import os
from typing import List
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
load_dotenv()

## TODO：https://docs.langchain.com/oss/python/langchain/sql-agent


def _get_db_uri() -> str:
    uri = os.getenv("RELATIONAL_DATABASE_URI")
    if uri:
        return uri
    user = os.getenv("RELATIONAL_DATABASE_USERNAME", "yuzhejun")
    password = os.getenv("RELATIONAL_DATABASE_PASSWORD", "mypassword")
    host = os.getenv("RELATIONAL_DATABASE_HOST", "localhost")
    port = os.getenv("RELATIONAL_DATABASE_PORT", "5432")
    name = os.getenv("RELATIONAL_DATABASE_NAME", "testyu")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"



def relational_database_retrieval_node(state: GlobalState):
    """
    Relational database retrieval node.
    """

    db = SQLDatabase.from_uri(_get_db_uri())

    model = ChatOpenAI(temperature=0, 
                    model_name="deepseek-chat",
                    api_key=os.getenv('DEEPSEEK_API_KEY'),
                    base_url="https://api.deepseek.com",
                    max_tokens=4000)

    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()
    
    system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)
    agent = create_sql_agent(
        llm=model,
        toolkit=toolkit,
        verbose=True,
        prefix=system_prompt,
    )

    documents: List[Document] = []
    multi_queries = state.get("multi_queries") or [state.get("original_question", "")]
    for query in multi_queries:
        if not query:
            continue
        try:
            result = agent.invoke({"input": query})
            output_text = result.get("output") if isinstance(result, dict) else str(result)
        except Exception as exc:
            output_text = f"SQL agent error: {exc}"
        documents.append(
            Document(
                page_content=output_text,
                metadata={"source": "relational_database", "query": query},
            )
        )

    return {"documents": documents}