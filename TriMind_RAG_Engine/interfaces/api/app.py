from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import time

from TriMind_RAG_Engine.services.rag_service import initialize_rag_pipeline

app = FastAPI(
    title="TriMind RAG API",
    version="1.0.0",
    description="Production-ready RAG backend"
)

logger = logging.getLogger(__name__)

# Global pipeline
rag_pipeline = None


# -------------------------------------------------
# Startup Event
# -------------------------------------------------
@app.on_event("startup")
def startup_event():
    global rag_pipeline
    logger.info("Initializing RAG pipeline at startup...")
    rag_pipeline = initialize_rag_pipeline()
    logger.info("RAG pipeline ready.")


# -------------------------------------------------
# Request Model
# -------------------------------------------------
class QueryRequest(BaseModel):
    question: str


# -------------------------------------------------
# Response Model
# -------------------------------------------------
class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]


# -------------------------------------------------
# Health Check
# -------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "RAG API running 🚀"}


# -------------------------------------------------
# Normal Ask Endpoint
# -------------------------------------------------
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    response = rag_pipeline.run(request.question)

    return {
        "answer": response.get("answer"),
        "confidence": response.get("confidence", 0.0),
        "sources": [
            {
                "file": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A")
            }
            for doc in response.get("source_documents", [])
        ]
    }


# -------------------------------------------------
# Streaming Endpoint (for Streamlit)
# -------------------------------------------------
@app.post("/ask_stream")
def ask_stream(request: QueryRequest):

    response = rag_pipeline.run(request.question)
    answer = response.get("answer", "")

    def generate():
        for word in answer.split():
            yield word + " "
            time.sleep(0.03)   # typing effect

    return StreamingResponse(generate(), media_type="text/plain")