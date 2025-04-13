import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from ...services.milvus_service import MilvusService
from ...services.watsonx_service import WatsonxService
from ...utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class CarbonResearchInput(BaseModel):
    """Input schema for CarbonResearchTool."""
    query: str = Field(..., description="The search query for finding carbon footprint data.")

class CarbonResearchTool(BaseTool):
    name: str = "carbon_research"
    description: str = "Searches for carbon footprint data in the Milvus database using semantic search"
    args_schema: Type[BaseModel] = CarbonResearchInput
    
    def __init__(self, milvus_service: MilvusService, watsonx_service: WatsonxService):
        """Initialize the carbon research tool."""
        super().__init__()
        self._milvus_service = milvus_service
        self._watsonx_service = watsonx_service
    
    def _run(self, query: str) -> str:
        """Execute the tool to search for carbon data.
        
        Args:
            query: The search query
            
        Returns:
            The search results as a string
        """
        logger.info(f"🔎 Searching for carbon data with query: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self._watsonx_service.generate_embedding(
                query,
                model_type="granite"
            )
            
            if not query_embedding:
                return "Failed to generate embedding for the query."
            
            # Search in Milvus
            results = self._milvus_service.search_vectors(
                collection_name="carbon_embeddings_granite",
                query_embedding=query_embedding,
                top_k=5
            )
            
            if not results:
                return "No relevant carbon data found."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(f"Result {i}:")
                formatted_results.append(f"Score: {result['score']:.4f}")
                formatted_results.append(f"Source: {result['source_file']}")
                formatted_results.append(f"Content: {result['text']}\n")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"❌ Error in CarbonResearchTool: {str(e)}", exc_info=True)
            return f"Error searching for carbon data: {str(e)}"