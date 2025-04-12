import logging
from typing import List, Dict, Any
from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService

logger = logging.getLogger(__name__)

class RAGGenerator:
    """Main class for RAG-based question answering."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the RAG generator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.milvus = MilvusService(config)
        self.watsonx = WatsonxService(config)
        logger.info("RAGGenerator initialized successfully")
    
    def generate_answer(self, query: str, collection_name: str = "carbon_embeddings_granite", top_k: int = 5) -> str:
        """Generate an answer using RAG pipeline.
        
        Args:
            query: User's question
            collection_name: Name of the Milvus collection
            top_k: Number of context chunks to retrieve
            
        Returns:
            Generated answer
        """
        try:
            # Step 1: Embed the query using the granite model
            query_vector = self.watsonx.generate_embedding(query, model_type="granite")
            if not query_vector:
                logger.error("Failed to generate query embedding")
                return "I apologize, but I encountered an error while processing your question. Please try again later."
            
            logger.debug("Query embedding generated successfully")
            
            # Step 2: Search Milvus
            results = self.milvus.search_vectors(collection_name, query_vector, top_k)
            logger.debug(f"Milvus search completed, found {len(results)} results")
            
            # Step 3: Gather context
            context_chunks = []
            for result in results:
                chunk = result.get("text")
                file_name = result.get("source_file")
                if chunk and file_name:
                    context_chunks.append(f"From {file_name}:\n{chunk.strip()}")
            
            if not context_chunks:
                logger.warning("No relevant context found in the search results")
                return "I cannot find information about that in the available data."
            
            context = "\n\n".join(context_chunks)
            logger.info(f"Retrieved {len(context_chunks)} relevant chunks from Milvus")
            
            # Step 4: Generate response
            prompt = self._construct_rag_prompt(query, context)
            
            # Define text generation parameters
            params = {
                "MAX_NEW_TOKENS": 2000,  # Maximum length of generated text
                "MIN_NEW_TOKENS": 100,  # Minimum length to ensure complete answers
                "TEMPERATURE": 0.7,     # Controls randomness (0.0 to 1.0)
                "TOP_P": 0.9,          # Nucleus sampling parameter
                "TOP_K": 50,           # Number of highest probability tokens to consider
                # "REPETITION_PENALTY": 1.2,  # Penalty for repeated tokens
                # "LENGTH_PENALTY": {    # Controls length of generated text
                #     "start_index": 50,
                #     "decay_factor": 1.2
                # },
                "STOP_SEQUENCES": ["\n\n", "Sources:", "References:"],  # Stop generation at these sequences
                "DECODING_METHOD": "greedy"  # Use greedy decoding for more focused responses
            }
            
            response = self.watsonx.generate_text(
                prompt=prompt,
                params=params
                # guardrails=True  # Enable content moderation
            )
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_answer: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while processing your question. Please try again later."
    
    def get_context(self, query: str, collection_name: str = "carbon_embeddings_granite", top_k: int = 5) -> List[Dict[str, str]]:
        """Get context chunks for a query without generating an answer.
        
        Args:
            query: User's question
            collection_name: Name of the Milvus collection
            top_k: Number of context chunks to retrieve
            
        Returns:
            List of context chunks with metadata
        """
        try:
            # Generate query embedding using the granite model
            query_vector = self.watsonx.generate_embedding(query, model_type="granite")
            if not query_vector:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search Milvus
            results = self.milvus.search_vectors(collection_name, query_vector, top_k)
            
            # Format results
            context_chunks = []
            for result in results:
                chunk = result.get("text")
                file_name = result.get("source_file")
                if chunk and file_name:
                    context_chunks.append({
                        "text": chunk.strip(),
                        "file_name": file_name,
                        "score": result.get("score", 0.0)
                    })
            
            return context_chunks
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}", exc_info=True)
            return []
    
    def _construct_rag_prompt(self, query: str, context: str) -> str:
        """Construct a prompt for the RAG pipeline.
        
        Args:
            query: User's question
            context: Retrieved context from Milvus
            
        Returns:
            Formatted prompt string
        """
        return f"""
        Based on the following context, please answer the question: {query}
    
        Context:
        {context}
        
        Please provide a clear and concise answer that:
        1. Directly addresses the question
        2. Uses information from the provided context
        3. Cites specific sources when possible
        4. Indicates if the information is incomplete or uncertain
        
        Keep your response focused and to the point.
        
        If the context does not contain relevant information, please state that clearly.
        """ 