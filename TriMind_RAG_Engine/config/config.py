import os
from dotenv import load_dotenv

# =========================================================
# Load Environment Variables
# =========================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================================================================================
# Data Configuration
DATA_PATH = "dataset"  # Change if needed
VECTORSTORE_PATH = "vectorstore"

# Embedding Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Configuration
PRIMARY_LLM = "gemini-2.5-flash"
PRIMARY_TEMPERATURE = 0.2

# Chunking Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retrieval Configuration
TOP_K = 10
SIMILARITY_THRESHOLD = 0.60

# Memory Configuration
MEMORY_WINDOW = 5
MEMORY_MAX_TOKENS = 2000

# Pinecone Configuration
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ai-engineering-hub")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# ===============================================================================================================================================