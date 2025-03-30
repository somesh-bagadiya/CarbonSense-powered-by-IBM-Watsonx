import os
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Milvus configuration
MILVUS_HOST = os.getenv("MILVUS_GRPC_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_GRPC_PORT"))
COS_API_KEY = os.getenv("COS_API_KEY")

def check_collection_data():
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            user="ibmlhapikey",
            password=COS_API_KEY,
            secure=True
        )
        logging.info("Successfully connected to Milvus")

        # Get the collection
        collection_name = "carbon_embeddings"
        if not utility.has_collection(collection_name):
            logging.error(f"Collection '{collection_name}' does not exist")
            return

        collection = Collection(name=collection_name)
        collection.load()
        
        # Get collection statistics
        row_count = collection.num_entities
        logging.info(f"Collection statistics:")
        logging.info(f"Number of entities: {row_count}")
        
        # If there's data, show a sample
        if row_count > 0:
            results = collection.query(
                expr="id >= 0",
                output_fields=["id", "file_name", "chunk_text"],
                limit=3
            )
            logging.info("\nSample data:")
            for result in results:
                logging.info(f"ID: {result['id']}")
                logging.info(f"File: {result['file_name']}")
                logging.info(f"Chunk: {result['chunk_text'][:100]}...")
                logging.info("-" * 50)

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    check_collection_data() 