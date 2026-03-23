from langchain_classic.chains import ConversationalRetrievalChain
from TriMind_RAG_Engine.memory.conversation_memory import get_conversation_memory
from TriMind_RAG_Engine.prompts.rag_prompt_template import CONVERSATIONAL_PROMPT

def create_conversational_chain(llm, retriever):
    """
    Creates a Conversational RAG chain with:
    - short-term memory
    - custom prompt
    - better response
    """

    # Create memory
    memory = get_conversation_memory(window_size=5)

    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt":CONVERSATIONAL_PROMPT,   # giving controlled response
            # "document_varible_name": "context"
            # Custom Domain Aware prompt is being used 
        }
    )

    return chain