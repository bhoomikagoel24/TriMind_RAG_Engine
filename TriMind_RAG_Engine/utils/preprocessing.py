import sys
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from TriMind_RAG_Engine.config.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException


logger = get_logger(__name__)


# Chunk Documents
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    """

    try:
        logger.info("Starting document chunking...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n## ",
                "\n### ",
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = text_splitter.split_documents(documents)

        # Add metadata enhancements
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["chunk_length"] = len(chunk.page_content)

        logger.info(f"Created {len(chunks)} chunks successfully")

        return chunks

    except Exception as e:
        logger.error("Error during chunking")
        raise CustomException(str(e), sys)

# Covering The Concepts
# ✔ Smart splitting
# ✔ Chunk overlap for context
# ✔ Metadata enrichment
# ✔ Logging
# ✔ Exception handling
# ✔ Config driven