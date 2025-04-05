from typing import List, Dict, Any
import logging
from .milvus_service import MilvusService
from .watsonx_service import WatsonxService
from ..config.config_manager import ConfigManager

class RAGService:
    """Service for Retrieval-Augmented Generation using Milvus and Watsonx."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the RAG service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.milvus = MilvusService(config)
        self.watsonx = WatsonxService(config)
        
    def generate_response(self, query: str, max_chunks: int = 3) -> str:
        """Generate a response to a query using RAG.
        
        Args:
            query: The user's question
            max_chunks: Maximum number of relevant chunks to use
            
        Returns:
            Generated response based on retrieved context
        """
        try:
            # Generate embedding for the query
            query_embedding = self.watsonx.generate_embeddings([query])[0]
            
            # Search for relevant chunks
            results = self.milvus.search_vectors(
                "carbon_embeddings",
                query_embedding,
                top_k=max_chunks
            )
            
            # Extract and format context
            context = []
            for hit in results:
                text = hit.entity.get("chunk_text")
                source = hit.entity.get("file_name")
                context.append(f"[Source: {source}]\n{text}")
            
            context_str = "\n\n".join(context)
            
            # Create prompt and generate response
            prompt = self.watsonx.construct_rag_prompt(query, context_str)
            response = self.watsonx.generate_text(prompt)
            return response
            
        except Exception as e:
            logging.error(f"Error in RAG generation: {str(e)}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def get_sources(self, query: str, max_chunks: int = 3) -> List[Dict[str, Any]]:
        """Get the source documents used for answering a query.
        
        Args:
            query: The user's question
            max_chunks: Maximum number of relevant chunks to retrieve
            
        Returns:
            List of source documents with their relevance scores
        """
        try:
            query_embedding = self.watsonx.generate_embeddings([query])[0]
            results = self.milvus.search_vectors(
                "carbon_embeddings",
                query_embedding,
                top_k=max_chunks
            )
            
            sources = []
            for hit in results:
                sources.append({
                    'file': hit.entity.get("file_name"),
                    'text': hit.entity.get("chunk_text"),
                    'distance': hit.distance
                })
            return sources
            
        except Exception as e:
            logging.error(f"Error retrieving sources: {str(e)}")
            return []

    async def search_sources(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant sources using the query."""
        try:
            # Generate query embedding
            query_embedding = await self.watsonx.generate_embeddings([query])
            
            # Search in Milvus with enhanced parameters
            search_params = {
                "metric_type": "COSINE",  # Use cosine similarity
                "params": {
                    "nprobe": 16,  # Number of clusters to search
                    "radius": 0.8   # Search radius
                }
            }
            
            # Perform the search
            results = await self.milvus.search_vectors(
                query_embedding[0],
                top_k=top_k,
                search_params=search_params
            )
            
            # Process and format results
            sources = []
            for hit in results:
                source = {
                    "file_name": hit.entity.get("file_name", "Unknown"),
                    "relevance_score": 1 - hit.distance,  # Convert distance to relevance score
                    "content_preview": hit.entity.get("text", ""),
                    "keywords": hit.entity.get("keywords", []),
                    "has_numbers": hit.entity.get("has_numbers", False),
                    "has_units": hit.entity.get("has_units", False)
                }
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logging.error(f"Error searching sources: {str(e)}")
            raise 