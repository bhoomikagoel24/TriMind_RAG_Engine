import sys
from typing import List

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

from TriMind_RAG_Engine.config.config import EMBEDDING_MODEL_NAME
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException


logger = get_logger(__name__)

# Custom SentenceTransformer Embeddings Wrapper

class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain-compatible wrapper for SentenceTransformer.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")

        except Exception as e:
            logger.error("Error loading embedding model")
            raise CustomException(str(e), sys)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            raise CustomException(str(e), sys)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            raise CustomException(str(e), sys)

# Factory Function
def get_embeddings():
    """
    Returns embedding model instance.
    """
    return SentenceTransformerEmbeddings()
