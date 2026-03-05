import sys
import os

from TriMind_RAG_Engine.config.config import DATA_PATH
from TriMind_RAG_Engine.utils.document_loader import load_documents
from TriMind_RAG_Engine.utils.preprocessing import chunk_documents
from TriMind_RAG_Engine.utils.embeddings import get_embeddings
from TriMind_RAG_Engine.utils.vector_store import create_vector_store
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException


logger = get_logger(__name__)


def run_ingestion(store_type: str = "pinecone", force: bool = False):
    """
    Run one-time ingestion process:
    Load documents → Chunk → Embed → Create Vector Store
    """

    try:
        logger.info("Starting Document Ingestion Process")

        # 1️⃣ Load documents
        documents = load_documents(DATA_PATH)
        logger.info(f"Loaded {len(documents)} documents.")

        if not documents:
            raise ValueError("No documents found in dataset folder.")

        # 2️⃣ Chunk documents
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks.")

        # 3️⃣ Load embeddings
        embeddings = get_embeddings()

        # 4️⃣ Create vector store
        create_vector_store(
            chunks=chunks,
            embeddings=embeddings,
            store_type=store_type,
            force=force
        )

        logger.info("Ingestion completed successfully.")

    except Exception as e:
        logger.error("Ingestion failed.")
        raise CustomException(str(e), sys)