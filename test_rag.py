from carbonsense.config.config_manager import ConfigManager
from carbonsense.services.rag_service import RAGService
import logging

def test_rag():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize services
    config = ConfigManager()
    rag = RAGService(config)
    
    # Test queries
    test_queries = [
        "What is the carbon footprint of dairy production in the USA?",
        "Compare the environmental impact of electricity production in different regions.",
        "What are the main sources of emissions in food manufacturing?",
        "How does the carbon footprint of electricity production vary across regions?",
        "What are the key factors affecting carbon emissions in the manufacturing sector?"
    ]
    
    for query in test_queries:
        logging.info(f"\n\nQuestion: {query}")
        
        # Get sources first
        sources = rag.get_sources(query)
        logging.info("\nRelevant Sources:")
        for i, source in enumerate(sources, 1):
            logging.info(f"\nSource {i}:")
            logging.info(f"File: {source['file']}")
            logging.info(f"Relevance Score: {1 - source['distance']:.2f}")
            logging.info(f"Content Preview: {source['text'][:200]}...")
        
        # Generate response
        logging.info("\nGenerating Response...")
        response = rag.generate_response(query)
        logging.info(f"\nResponse: {response}")

if __name__ == "__main__":
    test_rag() 