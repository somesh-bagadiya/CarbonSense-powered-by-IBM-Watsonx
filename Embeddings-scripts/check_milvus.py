import os
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Milvus configuration
MILVUS_HOST = os.getenv("MILVUS_GRPC_HOST")
MILVUS_PORT = int(os.getenv("MILVUS_GRPC_PORT"))
COS_API_KEY = os.getenv("COS_API_KEY")

def check_and_create_collection():
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

        # Try to get the collection
        try:
            collection = Collection(name="carbon_embeddings")
            logging.info("Collection 'carbon_embeddings' exists")
            collection.load()
            logging.info("Collection loaded successfully")
            return True
        except Exception as e:
            logging.warning(f"Collection 'carbon_embeddings' not found: {e}")
            
            # Create the collection
            logging.info("Creating new collection 'carbon_embeddings'")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500)
            ]
            schema = CollectionSchema(fields, description="CarbonSense document embeddings")
            collection = Collection(name="carbon_embeddings", schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            )
            collection.load()
            logging.info("Collection created and loaded successfully")
            return True

    except Exception as e:
        logging.error(f"Error: {e}")
        return False
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    check_and_create_collection() 