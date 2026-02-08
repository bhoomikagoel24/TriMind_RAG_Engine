"""
Configuration for AI/DS/MCP RAG System
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Project configuration"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Paths
    DATA_DIR = "rag_project/data"
    PDF_DIR = f"{DATA_DIR}/pdfs"
    PROCESSED_DIR = f"{DATA_DIR}/processed"
    
    # Model Settings
    LLM_MODEL = "llama-3.3-70b-versatile"  # Groq
    TEMPERATURE = 0.2
    
    # Embedding Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Vector Store
    INDEX_NAME = "ai-ds-mcp-knowledge"
    CLOUD = "aws"
    REGION = "us-east-1"
    
    # Chunking
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    
    # Retrieval
    INITIAL_K = 10
    FINAL_K = 4
    
    # Memory
    MEMORY_WINDOW = 5
    
    # Logging
    LOG_LEVEL = "INFO"