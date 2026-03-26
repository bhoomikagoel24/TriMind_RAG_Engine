import sys
from typing import Dict, Any
import re
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

# =========================================================
#  LLM QUERY REWRITING FUNCTION 
# =========================================================

def rewrite_query_llm(question: str, llm):
    """
    Rewrite user query into a concise, domain-aware search query
    (AI / MCP / Data Science focused)
    """

    try:
        prompt = f"""
        You are a query rewriting assistant specialized in AI, MCP, and Data Science.

        Convert the user question into a short and precise search query.

        Rules:
        - Keep meaning same
        - Focus on AI / MCP / Data Science context
        - Preserve important technical keywords (RAG, MCP, LLM, embeddings, etc.)
        - Make it concise and searchable
        - Return ONLY the rewritten query (no extra text)

        Examples:
        "what is rag in ai" → "rag in ai explanation"
        "define ai agent" → "ai agent definition"
        "rag vs fine tuning" → "rag vs fine tuning comparison"
        "how to use mcp in ai systems" → "mcp in ai systems usage"
        "applications of ai in data science" → "ai applications in data science"
        "how to learn python for data science" → "learn python for data science"

        Now rewrite:

        Question: {question}
        """

        # Call LLM (temperature low for consistency)
        response = llm.invoke(prompt)

        result = response.content.strip()

        # 🧹 Clean unwanted prefixes if model adds them
        if "Rewritten:" in result:
            result = result.split("Rewritten:")[-1].strip()

        if "Query:" in result:
            result = result.split("Query:")[-1].strip()

        # 🚫 Fallback if empty
        if not result:
            return question

        return result

    except Exception as e:
        print(f"[rewrite_query_llm ERROR]: {e}")
        return question
    
# =============================== RAG PIPELINE ==================================================
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

            # -------------------------------
            # SMALL TALK HANDLING
            # -------------------------------
            small = question.strip().lower()

            if small in ["ok", "okay", "thanks", "thank you"]:
                return {
                    "answer": "👍 Glad I could help! Ask anything else.",
                    "source_documents": [],
                    "confidence": 1.0,
                    "grounding_score": 1.0
                }

            if small in ["bye", "goodbye"]:
                return {
                    "answer": "👋 Goodbye! Feel free to return anytime.",
                    "source_documents": [],
                    "confidence": 1.0,
                    "grounding_score": 1.0
                }

            if len(question.split()) < 2:
                return {
                    "answer": "Please ask a more detailed question.",
                    "source_documents": [],
                    "confidence": 0.0,
                    "grounding_score": 0.0
                }
            # ---------------------------------------------------------------
            # Normalize question for cache => prevents duplicate cache keys
            # ---------------------------------------------------------------
          
            normalized_question = re.sub(r"[^\w\s]", "", question.strip().lower())

            cached = self.cache.get(normalized_question)
            if cached:
                return cached
            
            # --------------------------------------------
            # Call Conversational Chain
            # --------------------------------------------
            # response = self.chain.invoke({"question": question})
             
            llm = self.chain.combine_docs_chain.llm_chain.llm
            # STEP 1: rewrite using LLM
            rewritten_query = rewrite_query_llm(question, llm)

            logger.info(f"Original: {question}")
            logger.info(f"Rewritten: {rewritten_query}")

            # STEP 2: pass rewritten query
            hybrid_query = rewritten_query

            response = self.chain.invoke({
                "question": hybrid_query
            })

            answer = response.get("answer", "")
            docs = response.get("source_documents", [])

            # --------------------------------------------
            # Post-process retrieved documents
            # --------------------------------------------
            cleaned_docs = process_documents(rewritten_query, docs)

            # --------------------------------------------
            # Evaluation
            # --------------------------------------------
            evaluate_retrieval(rewritten_query,cleaned_docs)
            
            grounding_score = compute_grounding_score(answer, cleaned_docs)

            confidence = compute_confidence_score(answer, cleaned_docs, grounding_score)
            # print(f"Confidence Score: {confidence}")

            if confidence < 0.4:
                answer = (
                    "⚠️ The answer may be incomplete based on available documents.\n\n"
                    + answer
                )
           # --------------------------------------------
            # Final Response
            # --------------------------------------------
            final_response = {
                "answer": answer,
                "source_documents": cleaned_docs,
                "confidence": confidence,
                "grounding_score": grounding_score
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
        