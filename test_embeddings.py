from carbonsense.config.config_manager import ConfigManager
from carbonsense.services.milvus_service import MilvusService
from carbonsense.services.watsonx_service import WatsonxService
import logging

def test_semantic_search():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize services
    config = ConfigManager()
    milvus = MilvusService(config)
    watsonx = WatsonxService(config)
    
    # Test queries
    test_queries = [
        "What is the carbon footprint of electricity production in the USA?",
        "Tell me about the environmental impact of food production globally",
        "What are the emissions from dairy products?",
        "Compare renewable energy sources in different regions"
    ]
    
    try:
        for query in test_queries:
            logging.info(f"\nSearching for: {query}")
            
            # Generate embedding for the query
            query_embedding = watsonx.generate_embeddings([query])[0]
            
            # Search in Milvus
            results = milvus.search_vectors(
                "carbon_embeddings",
                query_embedding,
                top_k=3
            )
            
            logging.info("Top 3 relevant chunks:")
            for i, result in enumerate(results):
                logging.info(f"\nResult {i+1}:")
                logging.info(f"Distance: {result.distance}")
                logging.info(f"File: {result.entity.get('file_name')}")
                logging.info(f"Text: {result.entity.get('chunk_text')}")
                
    except Exception as e:
        logging.error(f"Error during semantic search: {str(e)}")

if __name__ == "__main__":
    test_semantic_search() 