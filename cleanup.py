import os
import logging
from dotenv import load_dotenv
from pymilvus import connections, Collection
import ibm_boto3
from ibm_botocore.client import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanupManager:
    """Manages cleanup of stored files and embeddings."""
    
    def __init__(self):
        """Initialize the cleanup manager."""
        load_dotenv(override=True)
        self._validate_environment()
        self._init_clients()
        
    def _validate_environment(self) -> None:
        """Validates required environment variables."""
        required_vars = {
            "COS_API_KEY": "IBM Cloud Object Storage API Key",
            "COS_INSTANCE_ID": "IBM Cloud Object Storage Instance ID",
            "COS_ENDPOINT": "IBM Cloud Object Storage Endpoint",
            "BUCKET_NAME": "IBM Cloud Object Storage Bucket Name",
            "MILVUS_GRPC_HOST": "Milvus GRPC Host",
            "MILVUS_GRPC_PORT": "Milvus GRPC Port",
            "MILVUS_CERT_PATH": "Milvus Certificate Path"
        }
        
        missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _init_clients(self) -> None:
        """Initialize COS and Milvus clients."""
        # Initialize COS client
        self.cos_client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=os.getenv("COS_API_KEY"),
            ibm_service_instance_id=os.getenv("COS_INSTANCE_ID"),
            config=Config(signature_version="oauth"),
            endpoint_url=os.getenv("COS_ENDPOINT")
        )
        
        # Initialize Milvus connection
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_GRPC_HOST"),
            port=os.getenv("MILVUS_GRPC_PORT"),
            user="ibmlhapikey",
            password=os.getenv("COS_API_KEY"),
            secure=True,
            server_ca=os.getenv("MILVUS_CERT_PATH")
        )
    
    def cleanup_cos_bucket(self) -> None:
        """Delete all files from the COS bucket."""
        try:
            bucket_name = os.getenv("BUCKET_NAME")
            
            # List all objects in the bucket
            objects = self.cos_client.list_objects_v2(Bucket=bucket_name)
            
            if 'Contents' in objects:
                # Create list of objects to delete
                delete_list = {'Objects': [{'Key': obj['Key']} for obj in objects['Contents']]}
                
                # Delete objects
                if delete_list['Objects']:
                    self.cos_client.delete_objects(
                        Bucket=bucket_name,
                        Delete=delete_list
                    )
                    logger.info(f"Deleted {len(delete_list['Objects'])} files from COS bucket")
                else:
                    logger.info("No files found in COS bucket")
            else:
                logger.info("Bucket is empty")
                
        except Exception as e:
            logger.error(f"Error cleaning up COS bucket: {str(e)}")
            raise
    
    def cleanup_milvus(self) -> None:
        """Drop all collections from Milvus."""
        try:
            # Get list of collections
            from pymilvus import utility
            collections = utility.list_collections()
            
            # Drop each collection
            for collection_name in collections:
                collection = Collection(name=collection_name)
                collection.drop()
                logger.info(f"Dropped Milvus collection: {collection_name}")
                
            if not collections:
                logger.info("No collections found in Milvus")
                
        except Exception as e:
            logger.error(f"Error cleaning up Milvus: {str(e)}")
            raise

    def cleanup_local_embeddings(self) -> None:
        """Delete all locally stored embeddings."""
        try:
            embeddings_dir = "Embeddings"
            if os.path.exists(embeddings_dir):
                # Delete all files in the Embeddings directory
                deleted_count = 0
                for file_name in os.listdir(embeddings_dir):
                    file_path = os.path.join(embeddings_dir, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                
                logger.info(f"Deleted {deleted_count} local embedding files")
            else:
                logger.info("Embeddings directory not found")
                
        except Exception as e:
            logger.error(f"Error cleaning up local embeddings: {str(e)}")
            raise
    
    def cleanup_all(self) -> None:
        """Clean up both COS bucket and Milvus collections."""
        try:
            logger.info("Starting cleanup process...")
            
            # Clean up COS bucket
            logger.info("Cleaning up COS bucket...")
            self.cleanup_cos_bucket()
            
            # Clean up Milvus
            logger.info("Cleaning up Milvus collections...")
            self.cleanup_milvus()

            # Clean up local embeddings
            logger.info("Cleaning up local embeddings...")
            self.cleanup_local_embeddings()
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
        finally:
            # Close Milvus connection
            connections.disconnect("default")

if __name__ == "__main__":
    try:
        cleanup_manager = CleanupManager()
        cleanup_manager.cleanup_all()
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        exit(1) 