import sys
from typing import Dict, Any

from TriMind_RAG_Engine.utils.document_processing import process_documents
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException
# Evaluation
from TriMind_RAG_Engine.utils.evaluation import evaluate_retrieval
# Evaluation Grounding for hsllucination check
from TriMind_RAG_Engine.utils.evaluation import compute_grounding_score
# computing confidence score
from TriMind_RAG_Engine.utils.evaluation import compute_confidence_score
# For caching
from TriMind_RAG_Engine.utils.cache import SimpleCache

logger = get_logger(__name__)


class RAGPipeline:
    """
    Central RAG Pipeline:
    Retriever → Clean → Memory → LLM
    """

    def __init__(self, conversational_chain):
        """
        Args:
            conversational_chain: ConversationalRetrievalChain instance
        """
        self.chain = conversational_chain
        self.cache = SimpleCache()
        logger.info("RAG Pipeline initialized successfully.")

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute full RAG pipeline

        Args:
            question (str): User input

        Returns:
            dict: Answer + source documents
        """

        try:
            logger.info(f"Processing question: {question}")

            # ---------------------------------------------------------------
            # Normalize question for cache => prevents duplicate cache keys
            # ---------------------------------------------------------------
            normalized_question = question.strip().lower()

            cached = self.cache.get(normalized_question)
            if cached:
                return cached
            
            # --------------------------------------------
            # Call Conversational Chain
            # --------------------------------------------
            response = self.chain.invoke({"question": question})

            answer = response.get("answer", "")
            docs = response.get("source_documents", [])

            # --------------------------------------------
            # Post-process retrieved documents
            # --------------------------------------------
            cleaned_docs = process_documents(question, docs)

            # --------------------------------------------
            # Evaluation
            # --------------------------------------------
            evaluate_retrieval(question,docs)
            
            compute_grounding_score(answer, docs)

            confidence = compute_confidence_score(answer, docs)
            print(f"Confidence Score: {confidence}")

           # --------------------------------------------
            # Final Response
            # --------------------------------------------
            final_response = {
                "answer": answer,
                "source_documents": cleaned_docs,
                "confidence": confidence
            }

            # --------------------------------------------
            # Store in cache (store FINAL result)
            # --------------------------------------------
            self.cache.set(normalized_question, final_response)

            logger.info("RAG pipeline execution completed.")

            return final_response
        
        except Exception as e:
            logger.error("RAG Pipeline failed.")
            raise CustomException(str(e), sys)