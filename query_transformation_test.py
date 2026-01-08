import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["DEEPSEEK_API_KEY"] = os.getenv('DEEPSEEK_API_KEY')


# Function for rewriting a query to improve retrieval
def rewrite_query(original_query, llm_chain):
    """
    Rewrite the original query to improve retrieval.

    Args:
    original_query (str): The original user query
    llm_chain: The chain used to generate the rewritten query

    Returns:
    str: The rewritten query
    """
    response = llm_chain.invoke(original_query)
    return response.content

def generate_multi_query(original_query, llm_chain):
    """
    Generate a multi-query to retrieve relevant information.

    Args:
    original_query (str): The original complex query
    llm_chain: The chain used to generate sub-queries

    Returns:
    List[str]: A list of queries 
    """
    response = llm_chain.invoke(original_query)
    multi_queries = [q.strip() for q in response.content.split('\n') if q.strip() and not q.strip().startswith('Multi-queries:')]
    return multi_queries

# Function for generating a step-back query to retrieve broader context
def generate_step_back_query(original_query, llm_chain):
    """
    Generate a step-back query to retrieve broader context.

    Args:
    original_query (str): The original user query
    llm_chain: The chain used to generate the step-back query

    Returns:
    str: The step-back query
    """
    response = llm_chain.invoke(original_query)
    return response.content


# Function for decomposing a query into simpler sub-queries
def decompose_query(original_query, llm_chain):
    """
    Decompose the original query into simpler sub-queries.

    Args:
    original_query (str): The original complex query
    llm_chain: The chain used to generate sub-queries

    Returns:
    List[str]: A list of simpler sub-queries
    """
    response = llm_chain.invoke(original_query).content
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    return sub_queries

# Main class for the RAG method
class RAGQueryProcessor:
    def __init__(self):
        # Initialize LLM models
        self.re_write_llm = ChatOpenAI(temperature=0, model_name="deepseek-chat",api_key=os.getenv('DEEPSEEK_API_KEY'),base_url="https://api.deepseek.com", max_tokens=4000)
        self.multi_query_llm = ChatOpenAI(temperature=0, model_name="deepseek-chat",api_key=os.getenv('DEEPSEEK_API_KEY'),base_url="https://api.deepseek.com", max_tokens=4000)
        self.step_back_llm = ChatOpenAI(temperature=0, model_name="deepseek-chat",api_key=os.getenv('DEEPSEEK_API_KEY'),base_url="https://api.deepseek.com", max_tokens=4000)
        self.sub_query_llm = ChatOpenAI(temperature=0, model_name="deepseek-chat",api_key=os.getenv('DEEPSEEK_API_KEY'),base_url="https://api.deepseek.com", max_tokens=4000)

        # Initialize prompt templates
        query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
        Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

        Original query: {original_query}

        Rewritten query:"""


        multi_query_template = """You are an AI language model assistant. Your task is to generate three different versions of the given user question 
        to retrieve relevant documents from a vector database.
        By generating multiple versions of the user question,
        your goal is to help the user overcome some of the limitations
        of distance-based similarity search. Provide these alternative questions separated by newlines.
        Original query: {original_query}

        Multi-queries:"""

        step_back_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.

        Original query: {original_query}

        Step-back query:"""

        subquery_decomposition_template = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

        Original query: {original_query}

        Example: What are the impacts of climate change on the environment?

        Example Sub-queries (without numbering):
        What are the impacts of climate change on biodiversity?
        How does climate change affect the oceans?
        What are the effects of climate change on agriculture?
        What are the impacts of climate change on human health?
        
        Sub-queries:"""

        # Create LLMChains
        self.query_rewriter = PromptTemplate(input_variables=["original_query"],
                                             template=query_rewrite_template) | self.re_write_llm
        self.multi_query_chain = PromptTemplate(input_variables=["original_query"],
                                                template=multi_query_template) | self.multi_query_llm
        self.step_back_chain = PromptTemplate(input_variables=["original_query"],
                                              template=step_back_template) | self.step_back_llm
        self.subquery_decomposer_chain = PromptTemplate(input_variables=["original_query"],
                                                        template=subquery_decomposition_template) | self.sub_query_llm

    def run(self, original_query):
        """
        Run the full RAG query processing pipeline.

        Args:
        original_query (str): The original query to be processed
        """
        # Rewrite the query
        rewritten_query = rewrite_query(original_query, self.query_rewriter)
        print("Original query:", original_query)
        print("\nRewritten query:", rewritten_query)

        # Generate multi-query
        multi_queries = generate_multi_query(original_query, self.multi_query_chain)
        print("\nMulti-queries:")
        for i, multi_query in enumerate(multi_queries, 1):
            print(f"{i}. {multi_query}")

        # Generate step-back query
        step_back_query = generate_step_back_query(original_query, self.step_back_chain)
        print("\nStep-back query:", step_back_query)

        # Decompose the query into sub-queries
        sub_queries = decompose_query(original_query, self.subquery_decomposer_chain)
        print("\nSub-queries:")
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"{i}. {sub_query}")


# Argument parsing
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Process a query using the RAG method.")
    parser.add_argument("--query", type=str, default='What are the impacts of climate change on the environment?',
                        help="The original query to be processed")
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    processor = RAGQueryProcessor()
    processor.run(args.query)