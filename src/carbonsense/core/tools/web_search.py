import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ...services.discovery_service import DiscoveryService
from ...utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class WebSearchInput(BaseModel):
    """Input schema for WebSearchTool."""
    query: str = Field(
        ...,
        description="The search query to find carbon footprint information"
    )

class WebSearchTool(BaseTool):
    """Tool for searching the web for carbon footprint information."""
    name: str = "web_search"
    description: str = "Searches the web for carbon footprint information and returns relevant results"
    args_schema: type[BaseModel] = WebSearchInput
    
    def __init__(self, discovery_service: DiscoveryService):
        """Initialize the web search tool."""
        super().__init__()
        self._discovery_service = discovery_service
    
    def _run(self, query: str) -> str:
        """Execute the tool to search the web."""
        logger.info(f"🌐 Searching web with query: {query}")
        
        try:
            search_query = f"carbon footprint {query}"
            results = self._discovery_service.search_web(search_query, max_results=5)
            
            if not results:
                return "No relevant web results found."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'Unknown')
                text = result.get('document_passages', [{}])[0].get(
                    'passage_text', result.get('text', '')
                )
                
                formatted_results.append(f"Web Result {i}:")
                formatted_results.append(f"Title: {title}")
                formatted_results.append(f"Content: {text}\n")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"❌ Error in WebSearchTool: {str(e)}", exc_info=True)
            return f"Error searching the web: {str(e)}"