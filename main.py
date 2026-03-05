# ========================================================================================================================================
import sys
import argparse

from TriMind_RAG_Engine.services.rag_service import initialize_rag_pipeline
from TriMind_RAG_Engine.services.ingestion_service import run_ingestion
from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException


logger = get_logger(__name__)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion process"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate vector index"
    )

    args = parser.parse_args()

    try:

        # ---------------- INGESTION MODE ----------------
        if args.ingest:
            run_ingestion(force=args.force)
            return

        # ---------------- RUNTIME MODE ----------------
        logger.info("Starting RAG Application")

        rag_pipeline = initialize_rag_pipeline()

        print("\nRAG Chatbot Ready! Type 'exit' to quit.\n")

        while True:
            query = input("You: ").strip()

            if query.lower() in ["exit", "quit","bye"]:
                print("Goodbye 👋")
                break

            if not query:
                continue

            response = rag_pipeline.run(query)

            print("\nAssistant:\n")
            print(response["answer"])

            if "source_documents" in response:
                print("\nSources:")
                for i, doc in enumerate(response["source_documents"][:2], 1):
                    print(f"{i}. {doc.metadata.get('source', 'Unknown')}")
            
            if "confidence" in response:
                print(f"\nConfidence: {response['confidence']}")
                
            print("\n" + "-" * 60)

    except Exception as e:
        logger.error("RAG pipeline failed.")
        raise CustomException(str(e), sys)


if __name__ == "__main__":
    main()

# ========================================================================================================================================