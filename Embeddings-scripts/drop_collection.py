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

def drop_collection():
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

        # Drop the collection
        collection_name = "carbon_embeddings"
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logging.info(f"Successfully dropped collection '{collection_name}'")
        else:
            logging.info(f"Collection '{collection_name}' does not exist")

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    drop_collection() 