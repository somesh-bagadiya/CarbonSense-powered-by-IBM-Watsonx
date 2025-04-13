def initialize_tools(config):
    """Initialize tools based on the provided configuration.
    
    Args:
        config: The configuration manager instance
        
    Returns:
        Dictionary of tool instances keyed by their names
    """
    from .carbon_research import CarbonResearchTool
    from .web_search import WebSearchTool
    from ...services.milvus_service import MilvusService
    from ...services.watsonx_service import WatsonxService
    from ...services.discovery_service import DiscoveryService
    
    tools = {}
    
    # Initialize the required services
    milvus_service = MilvusService(config)
    watsonx_service = WatsonxService(config)
    discovery_service = DiscoveryService(config)
    
    # Initialize CarbonResearchTool with the required services
    tools["carbon_research"] = CarbonResearchTool(milvus_service, watsonx_service)
    
    # Initialize WebSearchTool with the required service
    tools["web_search"] = WebSearchTool(discovery_service)
    
    return tools
