from global_state import GlobalState
from langchain_core.documents import Document
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import psycopg2
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agents import create_sql_agent
from langchain_community.tools import SQLDatabaseToolkit

load_dotenv()

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - fallback if psycopg2 isn't installed
    psycopg2 = None

try:
    import psycopg  # type: ignore
except Exception:  # pragma: no cover - fallback if psycopg isn't installed
    psycopg = None


## TODO：https://docs.langchain.com/oss/python/langchain/sql-agent
def _get_env_value(name: str, required: bool = True) -> Optional[str]:
    value = os.getenv(name)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _get_connection_params() -> Dict[str, Any]:
    return {
        "dbname": _get_env_value("RELATIONAL_DATABASE_NAME"),
        "user": _get_env_value("RELATIONAL_DATABASE_USERNAME"),
        "password": _get_env_value("RELATIONAL_DATABASE_PASSWORD"),
        "host": _get_env_value("RELATIONAL_DATABASE_HOST"),
        "port": _get_env_value("RELATIONAL_DATABASE_PORT"),
        "sslmode": os.getenv("RELATIONAL_DATABASE_SSLMODE"),
    }


def _connect() -> Any:
    params = _get_connection_params()
    if psycopg2 is not None:
        return psycopg2.connect(**params)
    if psycopg is not None:
        return psycopg.connect(**params)
    raise ImportError(
        "No PostgreSQL driver found. Install psycopg2-binary or psycopg."
    )


def _extract_sql_query(state: GlobalState) -> str:
    sql_query = (
        state.get("sql_query")
        or state.get("relational_query")
        or state.get("query")
        or ""
    )
    if not sql_query:
        raise ValueError(
            "No SQL query provided in state. Set state['sql_query']."
        )
    return sql_query


def _rows_to_documents(
    rows: List[Tuple[Any, ...]], columns: List[str]
) -> List[Document]:
    documents: List[Document] = []
    for row in rows:
        record = dict(zip(columns, row)) if columns else {"result": row}
        documents.append(
            Document(
                page_content=json.dumps(record, ensure_ascii=True, default=str),
                metadata={"source": "relational_database"},
            )
        )
    return documents

db = SQLDatabase.from_uri("postgresql+psycopg2://myuser:mypassword@localhost:5432/testyu")

llm = ChatOpenAI(temperature=0, 
                model_name="deepseek-chat",
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com",
                max_tokens=4000)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
def relational_database_retrieval_node(state: GlobalState):
    """
    Relational database retrieval node.
    """
    conn = psycopg2.connect(
    dbname="testyu",
    user="myuser",
    password="mypassword",
    host="localhost",
    port="5432"
)
    cursor = conn.cursor()


    def text_to_sql(text):
    # Use LangChain to generate SQL from text
        sql_query = langchain.text_to_sql(text)
        
        try:
            # Execute the SQL query
            cursor.execute(sql_query)
            # Fetch and return the results
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            return results
        except Exception as e:
            return str(e)

    documents = []
    # Example prompt to convert text to SQL
    multi_queries = state["multi_queries"]
    for query in multi_queries:
        results = text_to_sql(query)

        documents.append(results)
    
    cursor.close()
    conn.close()
    return {"documents": documents}