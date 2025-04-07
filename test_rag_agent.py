from carbonsense.config.config_manager import ConfigManager
from carbonsense.core.carbon_agent import CarbonAgent
import logging

def test_rag_agent():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize services
    config = ConfigManager()
    agent = CarbonAgent(config)
    
    # Test queries with varying complexity
    test_queries = [
        "What is carbon footprint of using 10 paper napkins?"
        # "Compare the emissions of electric vehicles vs gasoline vehicles",
        # "What are the most sustainable energy sources for data centers?",
        # "How does air travel impact carbon emissions globally?"
    ]
    
    try:
        for query in test_queries:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing query: {query}")
            
            # Process query with agent
            result = agent.process_query(query)
            
            # Log results
            logging.info("\nResponse:")
            logging.info(result["response"])
            
            if result["sources"]:
                logging.info("\n\nSources used:")
                for source in result["sources"]:
                    logging.info(f"- {source['file']} (Score: {source['score']:.2f})")
            
            if result["web_search_used"]:
                logging.info("\nWeb search was used to supplement the response")
            
            logging.info(f"{'='*50}\n")
            
    except Exception as e:
        logging.error(f"Error during RAG testing: {str(e)}")

if __name__ == "__main__":
    test_rag_agent() 