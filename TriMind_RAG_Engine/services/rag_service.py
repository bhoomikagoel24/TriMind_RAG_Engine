from langchain_google_genai import ChatGoogleGenerativeAI
from TriMind_RAG_Engine.chains.conversational_chain import create_conversational_chain
from TriMind_RAG_Engine.chains.rag_pipeline import RAGPipeline
from TriMind_RAG_Engine.config.config import (
    DATA_PATH,
    PRIMARY_LLM,
    PRIMARY_TEMPERATURE,
)

from TriMind_RAG_Engine.utils.document_loader import load_documents
from TriMind_RAG_Engine.utils.preprocessing import chunk_documents
from TriMind_RAG_Engine.utils.embeddings import get_embeddings
from TriMind_RAG_Engine.utils.vector_store import load_vector_store
from TriMind_RAG_Engine.chains.retrieval_chain import (
    create_advanced_retriever,
    create_rag_chain
)


def initialize_rag_pipeline():
    """
    Initializes full RAG system and returns ready chain
    """

    # Load documents
    documents = load_documents(DATA_PATH)

    # Chunk
    chunks = chunk_documents(documents)

    # Embeddings
    embeddings = get_embeddings()

    # Vector store
    vectorstore = load_vector_store(
        embeddings=embeddings,
        store_type="pinecone"
    )

    # LLM
    llm = ChatGoogleGenerativeAI(
        model=PRIMARY_LLM,
        temperature=PRIMARY_TEMPERATURE
    )

    # Retriever
    retriever = create_advanced_retriever(vectorstore)

    # Chain => [ CONVERSATIONAL CHAIN ]
    # rag_chain = create_conversational_chain(llm, retriever)
    # return rag_chain
    
    conversational_chain = create_conversational_chain(llm, retriever)
    pipeline = RAGPipeline(conversational_chain)
    
    return pipeline