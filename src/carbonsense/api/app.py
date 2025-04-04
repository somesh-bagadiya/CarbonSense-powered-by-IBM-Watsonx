from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from ..config.config_manager import ConfigManager
from ..core.rag_generator import RAGGenerator
from ..core.embedding_generator import EmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CarbonSense API",
    description="API for carbon footprint analysis using RAG and Watsonx",
    version="0.1.0"
)

class Query(BaseModel):
    text: str
    show_context: bool = False

class Answer(BaseModel):
    answer: str
    context: Optional[List[dict]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        app.state.config = ConfigManager()
        app.state.rag = RAGGenerator(app.state.config)
        app.state.embedding_generator = EmbeddingGenerator(app.state.config)
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.post("/query", response_model=Answer)
async def process_query(query: Query):
    """Process a user query and return the answer."""
    try:
        answer = app.state.rag.generate_answer(query.text)
        response = {"answer": answer}
        
        if query.show_context:
            context = app.state.rag.get_context(query.text)
            response["context"] = context
            
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-embeddings")
async def generate_embeddings():
    """Generate embeddings for all documents in COS."""
    try:
        app.state.embedding_generator.process_all_files()
        return {"message": "Embeddings generated successfully"}
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 