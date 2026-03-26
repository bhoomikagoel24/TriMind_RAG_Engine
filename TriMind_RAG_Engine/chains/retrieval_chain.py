import sys
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.chains import RetrievalQA

from TriMind_RAG_Engine.config.config import TOP_K
from TriMind_RAG_Engine.config.config import SIMILARITY_THRESHOLD
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException
# High-Quality chlunks -> better answers

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
        return vectorstore.as_retriever(
            search_type="mmr",                           # MMR = Max Marginal Relevance
            search_kwargs={"k": k,"fetch_k":20})         # Instead of Top 10 similar chunks It gives Top 10 relevant but diverse chunks
    except Exception as e:
        raise CustomException(str(e), sys)


# =========================================================
# ADVANCED RETRIEVER (Custom Scoring Logic)
# =========================================================

def create_advanced_retriever(vectorstore, k: int = TOP_K, threshold:float=SIMILARITY_THRESHOLD):
    """
    Custom retriever with scoring boost logic and similarity threshold filtering.
    """

    class EfficientRetriever(BaseRetriever):
        vectorstore: object
        k: int

        def _get_relevant_documents(self,query: str,**kwargs ) -> List[Document]:

            candidates = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self.k * 3
            )

            scored_docs = []
            query_lower = query.lower()

            filtered_docs = []

            for doc, base_score in candidates:

                # threshold filtering
                if base_score >= threshold:
                    filtered_docs.append((doc, base_score))

            # fallback (VERY IMPORTANT)
            if not filtered_docs:
                filtered_docs = candidates[:self.k]            
            
            for doc, base_score in filtered_docs:
                
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

                length_penalty = len(content_lower) / 1000
                final_score = base_score + boost - length_penalty
                scored_docs.append((doc, final_score))

            # Sort by boosted score
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in scored_docs[:self.k]]

    return EfficientRetriever(vectorstore=vectorstore, k=k, threshold= SIMILARITY_THRESHOLD)


# =========================================================
# CREATE RAG CHAIN
# =========================================================
from TriMind_RAG_Engine.utils.document_processing import process_documents

MAX_CONTEXT_LENGTH = 1500   # context windoe limit

def create_rag_chain(llm, retriever):

    class CustomRAGChain:

        def __init__(self, llm, retriever):
            self.llm = llm
            self.retriever = retriever

        def invoke(self, inputs: dict):

            query = inputs.get("query") or inputs.get("question")

            raw_docs = self.retriever.invoke(query)
            processed_docs = process_documents(query, raw_docs)

            # context control
            context_chunks = []
            total_length = 0

            for doc in processed_docs:
                chunk = doc.page_content

                if total_length + len(chunk) > MAX_CONTEXT_LENGTH:
                    break

                context_chunks.append(chunk)
                total_length += len(chunk)

            context = "\n\n".join(context_chunks)

            prompt = f"""
            You are an expert AI assistant.

            Use ONLY the provided context to answer.

            Instructions:
            - Give a clear and structured answer
            - Start with a short definition (if applicable)
            - Then explain step-by-step or in bullet points
            - Do NOT use outside knowledge
            - If answer is not in context → say "Not available in context"

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

            response = self.llm.invoke(prompt)

            return {
                "answer": response.content,
                "source_documents": processed_docs
            }

    logger.info("Custom RAG chain ready")

    return CustomRAGChain(llm, retriever)
