"""
Document loading utilities
"""
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader
)
from typing import List
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

def load_pdfs(pdf_dir: str) -> List[Document]:
    """
    Load all PDFs from directory
    
    Args:
        pdf_dir: Path to PDF directory
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading PDFs from: {pdf_dir}")
    
    try:
        loader = DirectoryLoader(
            pdf_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        docs = loader.load()
        logger.info(f"✅ Loaded {len(docs)} document pages")
        
        return docs
        
    except Exception as e:
        logger.error(f"❌ Error loading PDFs: {e}")
        return []

def filter_metadata(docs: List[Document]) -> List[Document]:
    """
    Keep only essential metadata
    
    Args:
        docs: List of documents
        
    Returns:
        Filtered documents
    """
    logger.info("Filtering metadata...")
    
    filtered = []
    for doc in docs:
        filtered.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": doc.metadata.get("source", "unknown")}
            )
        )
    
    logger.info(f"✅ Filtered {len(filtered)} documents")
    return filtered