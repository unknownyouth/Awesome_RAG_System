from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
import os
from dotenv import load_dotenv

load_dotenv()

db = SQLDatabase.from_uri("postgresql://yuzhejun:mypassword@localhost:5432/testyu")



print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")

llm = ChatOpenAI(temperature=0, 
                model_name="deepseek-chat",
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com",
                max_tokens=4000)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

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
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=system_prompt,
    )

print(agent.invoke({"input": "Tell me about the table ''test"}))