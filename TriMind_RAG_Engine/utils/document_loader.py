import os
import sys
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_core.documents import Document

from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException
from TriMind_RAG_Engine.config.config import DATA_PATH


logger = get_logger(__name__)

# Load All Documents (PDF + TXT)

def load_documents(data_path: str = DATA_PATH) -> List[Document]:
    """
    Load all supported documents from dataset folder.
    """

    try:
        logger.info(f"Loading documents from: {data_path}")

        # Load PDFs
        pdf_loader = DirectoryLoader(
            data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )

        # Load TXTs
        txt_loader = DirectoryLoader(
            data_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        pdf_docs = pdf_loader.load()
        txt_docs = txt_loader.load()

        all_docs = pdf_docs + txt_docs

        logger.info(f"Loaded {len(all_docs)} documents successfully")

        return all_docs

    except Exception as e:
        logger.error("Error while loading documents")
        raise CustomException(str(e), sys)


# Filter Minimal Metadata
def filter_minimal_metadata(docs: List[Document]) -> List[Document]:
    """
    Keep only essential metadata (source).
    """

    try:
        minimal_docs = []

        for doc in docs:
            src = doc.metadata.get("source")

            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": src}
                )
            )

        logger.info("Metadata filtered successfully")
        return minimal_docs

    except Exception as e:
        raise CustomException(str(e), sys)
