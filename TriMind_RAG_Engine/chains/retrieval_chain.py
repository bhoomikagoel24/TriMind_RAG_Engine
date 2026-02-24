import sys
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.chains import RetrievalQA

from TriMind_RAG_Engine.config.config import TOP_K
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException


logger = get_logger(__name__)


# =========================================================
# BASIC RETRIEVER
# =========================================================

def create_basic_retriever(vectorstore, k: int = TOP_K):
    """
    Standard similarity-based retriever
    """
    try:
        logger.info(f"Creating basic retriever with k={k}")
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        raise CustomException(str(e), sys)


# =========================================================
# ADVANCED RETRIEVER (Custom Scoring Logic)
# =========================================================

def create_advanced_retriever(vectorstore, k: int = TOP_K):
    """
    Custom retriever with scoring boost logic
    """

    class EfficientRetriever(BaseRetriever):
        vectorstore: object
        k: int

        def _get_relevant_documents(
            self,
            query: str,
            **kwargs
        ) -> List[Document]:

            candidates = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self.k * 3
            )

            scored_docs = []
            query_lower = query.lower()

            for doc, base_score in candidates:
                content_lower = doc.page_content.lower()
                boost = 0

                # Exact phrase boost
                if query_lower in content_lower:
                    boost += 0.2

                # Early position boost
                if any(term in content_lower[:100] for term in query_lower.split()):
                    boost += 0.1

                # Term coverage boost
                query_terms = query_lower.split()
                if query_terms:
                    coverage = sum(
                        1 for term in query_terms if term in content_lower
                    ) / len(query_terms)
                    boost += coverage * 0.1

                final_score = base_score + boost
                scored_docs.append((doc, final_score))

            # Sort by boosted score
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in scored_docs[:self.k]]

    logger.info(f"Creating advanced retriever with k={k}")
    return EfficientRetriever(vectorstore=vectorstore, k=k)


# =========================================================
# CREATE RAG CHAIN
# =========================================================

def create_rag_chain(llm, retriever):
    """
    Create RetrievalQA chain
    """

    try:
        logger.info("Initializing RetrievalQA chain")

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        logger.info("RAG chain ready")
        return chain

    except Exception as e:
        raise CustomException(str(e), sys)