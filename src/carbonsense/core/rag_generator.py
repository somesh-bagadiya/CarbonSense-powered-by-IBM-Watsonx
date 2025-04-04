import logging
from typing import List, Dict, Any
from ..config.config_manager import ConfigManager
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService

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
    
    def generate_answer(self, query: str, collection_name: str = "carbon_embeddings", top_k: int = 5) -> str:
        """Generate an answer using RAG pipeline.
        
        Args:
            query: User's question
            collection_name: Name of the Milvus collection
            top_k: Number of context chunks to retrieve
            
        Returns:
            Generated answer
        """
        try:
            # Step 1: Embed the query
            query_vector = self.watsonx.generate_embeddings([query])[0]
            logging.debug("Query embedding generated successfully")
            
            # Step 2: Search Milvus
            results = self.milvus.search_vectors(collection_name, query_vector, top_k)
            logging.debug(f"Milvus search completed, found {len(results)} results")
            
            # Step 3: Gather context
            context_chunks = []
            for hit in results:
                chunk = hit.entity.get("chunk_text")
                file_name = hit.entity.get("file_name")
                if chunk and file_name:
                    context_chunks.append(f"From {file_name}:\n{chunk.strip()}")
            
            if not context_chunks:
                logging.warning("No relevant context found in the search results")
                return "I cannot find information about that in the available data."
            
            context = "\n\n".join(context_chunks)
            logging.info(f"Retrieved {len(context_chunks)} relevant chunks from Milvus")
            
            # Step 4: Generate response
            prompt = self.watsonx.construct_rag_prompt(query, context)
            response = self.watsonx.generate_text(prompt)
            return response
            
        except Exception as e:
            logging.error(f"Error in generate_answer: {str(e)}", exc_info=True)
            return f"An error occurred while processing your question: {str(e)}"
    
    def get_context(self, query: str, collection_name: str = "carbon_embeddings", top_k: int = 5) -> List[Dict[str, str]]:
        """Get context chunks for a query without generating an answer.
        
        Args:
            query: User's question
            collection_name: Name of the Milvus collection
            top_k: Number of context chunks to retrieve
            
        Returns:
            List of context chunks with metadata
        """
        try:
            # Generate query embedding
            query_vector = self.watsonx.generate_embeddings([query])[0]
            
            # Search Milvus
            results = self.milvus.search_vectors(collection_name, query_vector, top_k)
            
            # Format results
            context_chunks = []
            for hit in results:
                chunk = hit.entity.get("chunk_text")
                file_name = hit.entity.get("file_name")
                if chunk and file_name:
                    context_chunks.append({
                        "text": chunk.strip(),
                        "file_name": file_name,
                        "score": hit.score
                    })
            
            return context_chunks
            
        except Exception as e:
            logging.error(f"Error getting context: {str(e)}")
            return [] 