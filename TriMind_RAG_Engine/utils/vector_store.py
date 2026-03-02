import sys
from typing import Literal

from langchain_community.vectorstores import FAISS, Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from TriMind_RAG_Engine.config.config import (
    VECTORSTORE_PATH,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
)

from TriMind_RAG_Engine.logging.logger import get_logger
from TriMind_RAG_Engine.exception_handler.custom_exception import CustomException


logger = get_logger(__name__)


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(
    chunks,
    embeddings,
    store_type: Literal["faiss", "chroma", "pinecone"] = "faiss",
    force:bool = False  # as if index exists it will reuse
):

    try:
        logger.info(f"Creating vector store: {store_type}")

        # ---------------- FAISS ----------------
        if store_type == "faiss":

            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            vectorstore.save_local(VECTORSTORE_PATH)

            logger.info(f"FAISS store created with {len(chunks)} vectors.")
            return vectorstore


        # ---------------- CHROMA ----------------
        elif store_type == "chroma":

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=VECTORSTORE_PATH
            )

            vectorstore.persist()

            logger.info(f"Chroma store created with {len(chunks)} vectors.")
            return vectorstore


        # ---------------- PINECONE ----------------
        elif store_type == "pinecone":

            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not set in .env")

            pc = Pinecone(api_key=PINECONE_API_KEY)

            # Force deleting existing index
            if force and pc.has_index(PINECONE_INDEX_NAME):
                logger.info(f"Force enabled. Deleting index: {PINECONE_INDEX_NAME}")
                pc.delete_index(PINECONE_INDEX_NAME)

            # Detect embedding dimension automatically
            sample_vector = embeddings.embed_query("test")
            dimension = len(sample_vector)

            # Create index if not exists
            if not pc.has_index(PINECONE_INDEX_NAME):

                logger.info(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=PINECONE_CLOUD,
                        region=PINECONE_REGION
                    )
                )

            else:
                logger.info(f"Using existing Pinecone index: {PINECONE_INDEX_NAME}")

            vectorstore = PineconeVectorStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                pinecone_api_key=PINECONE_API_KEY
            )

            logger.info(f"Pinecone store created with {len(chunks)} vectors.")
            return vectorstore


        else:
            raise ValueError("Unsupported vector store type.")

    except Exception as e:
        logger.error("Vector store creation failed.")
        raise CustomException(str(e), sys)


# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store(
    embeddings,
    store_type: Literal["faiss", "chroma", "pinecone"] = "faiss"
):

    try:
        logger.info(f"Loading vector store: {store_type}")

        # ---------------- FAISS ----------------
        if store_type == "faiss":

            return FAISS.load_local(
                VECTORSTORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )


        # ---------------- CHROMA ----------------
        elif store_type == "chroma":

            return Chroma(
                persist_directory=VECTORSTORE_PATH,
                embedding_function=embeddings
            )


        # ---------------- PINECONE ----------------
        elif store_type == "pinecone":

            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not set in .env")

            return PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY
            )


        else:
            raise ValueError("Unsupported vector store type.")

    except Exception as e:
        logger.error("Vector store loading failed.")
        raise CustomException(str(e), sys)