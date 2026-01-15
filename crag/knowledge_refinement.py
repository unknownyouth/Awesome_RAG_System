from global_state import GlobalState

def knowledge_refinement_node(state: GlobalState):
    '''
    Knowledge refinement node.
    '''
    knowledge = state["reranked_documents"]
    final_documents = knowledge

    return {"final_documents": final_documents}