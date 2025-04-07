import logging
from typing import List, Dict, Any
from ibm_watson import DiscoveryV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ..config.config_manager import ConfigManager

class DiscoveryService:
    """Service for interacting with IBM Watson Discovery for web search."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the Discovery service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.discovery_config = config.get_discovery_config()
        self.authenticator = IAMAuthenticator(self.discovery_config["api_key"])
        self.discovery = DiscoveryV2(
            version=self.discovery_config["version"],
            authenticator=self.authenticator
        )
        self.discovery.set_service_url(self.discovery_config["url"])
        self.project_id = self.discovery_config["project_id"]
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using Watson Discovery.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        try:
            # Prepare query parameters
            query_params = {
                "query": query,
                "count": max_results,
                "return_": ["text", "url", "title", "confidence"]
            }
            
            # Execute search
            response = self.discovery.query(
                project_id=self.project_id,
                **query_params
            ).get_result()
            
            # Format results
            results = []
            for result in response.get("results", []):
                results.append({
                    "text": result.get("text", " N/A "),
                    "document_passages": result.get("document_passages", " N/A "),
                    "title": result.get("title", " N/A "),
                    "document_id": result.get("document_id", " N/A "),
                    "result_metadata": result.get("result_metadata", " N/A ")
                })
            
            return results
            
        except Exception as e:
            logging.error(f"Error searching web: {str(e)}")
            return [] 