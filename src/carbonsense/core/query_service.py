import logging
from typing import Dict, Any
from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService

class QueryService:
    """Service for processing queries using RAG."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the query service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.milvus = MilvusService(config)
        self.watsonx = WatsonxService(config)
        self.collection_name = "carbon_embeddings_granite"  # Collection name used during embedding generation
    
    def process_query(self, query: str, show_context: bool = False) -> Dict[str, Any]:
        """Process a query and return relevant chunks with optional context."""
        try:
            logging.info("Processing query...")
            
            # Generate embedding for the query
            query_embedding = self.watsonx.generate_embedding(query, model_type="granite")
            if not query_embedding:
                raise ValueError("Failed to generate query embedding")
            
            # Search for relevant chunks
            results = self.milvus.search_vectors(self.collection_name, query_embedding, top_k=5)
            
            # Format results
            formatted_results = []
            for result in results:
                chunk = result['text']
                score = result['score']
                file_name = result['source_file']
                
                if show_context:
                    # Get context from the file
                    context = self._get_context_from_file(file_name, chunk)
                    formatted_results.append({
                        'chunk': chunk,
                        'score': score,
                        'file_name': file_name,
                        'context': context
                    })
                else:
                    formatted_results.append({
                        'chunk': chunk,
                        'score': score,
                        'file_name': file_name
                    })
            
            return {
                'query': query,
                'results': formatted_results
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

    def _get_context_from_file(self, file_name: str, chunk: str) -> str:
        # Implementation of _get_context_from_file method
        # This method should return the context for a given file and chunk
        # It's a placeholder and should be implemented based on your specific requirements
        return ""  # Placeholder return, actual implementation needed

    def _get_context_from_file(self, file_name: str, chunk: str) -> str:
        # Implementation of _get_context_from_file method
        # This method should return the context for a given file and chunk
        # It's a placeholder and should be implemented based on your specific requirements
        return ""  # Placeholder return, actual implementation needed 