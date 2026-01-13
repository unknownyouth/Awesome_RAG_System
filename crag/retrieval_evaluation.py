from global_state import GlobalState
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

class RetrievalEvaluator(BaseModel):
    """Classify retrieved documents based on how relevant it is to the user's question."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

llm = ChatOpenAI(temperature=0,
                 model_name="deepseek-chat",
                 api_key=os.getenv('DEEPSEEK_API_KEY'),
                 base_url="https://api.deepseek.com", 
                 max_tokens=4000)

def retrieval_evaluation_node(state: GlobalState):
    """
    Evaluate the retrieval performance.
    """

    rewritten_query = state["rewritten_query"]
    reranked_documents = state["reranked_documents"]
    retrieval_evaluation = []

    system_prompt = """You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question. \\n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \\n
    Output a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
    
    Retrieved document: {document}

    User's question: {rewritten_query}

    {format_instructions}
    """

    parser = PydanticOutputParser(pydantic_object=RetrievalEvaluator)
        
    format_instructions = parser.get_format_instructions()

    for document in reranked_documents:
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["document", "rewritten_query"],
            partial_variables={"format_instructions": format_instructions},
        )

        chain = prompt | llm | parser
    
        result = chain.invoke({"document": document, "rewritten_query": rewritten_query})
        retrieval_evaluation.append(result.binary_score)
    return {"retrieval_evaluation": retrieval_evaluation}