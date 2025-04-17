import os
import logging
from typing import Dict, Any
from pymilvus import connections, Collection, utility
import ibm_boto3
from ibm_botocore.client import Config
from ..config.config_manager import ConfigManager
from ..utils.logger import setup_logger

# Use the same logger setup as main
logger = setup_logger(__name__)

class CleanupService:
    """Service for cleaning up stored files and embeddings."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the cleanup service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self._init_clients()
        
    def _init_clients(self) -> None:
        """Initialize COS and Milvus clients."""
        # Initialize COS client
        cos_config = self.config.get_cos_config()
        self.cos_client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=cos_config["api_key"],
            ibm_service_instance_id=cos_config["instance_id"],
            config=Config(signature_version="oauth"),
            endpoint_url=cos_config["endpoint"]
        )
        
        # Initialize Milvus connection
        milvus_config = self.config.get_milvus_config()
        connections.connect(
            alias="default",
            host=milvus_config["host"],
            port=milvus_config["port"],
            user="ibmlhapikey",
            password=cos_config["api_key"],
            secure=True,
            server_ca=milvus_config["cert_path"]
        )
    
    def cleanup_cos_bucket(self) -> None:
        """Delete all files from the COS bucket."""
        try:
            bucket_name = self.config.get_cos_config()["bucket"]
            
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
            logger.info(f"Starting cleanup of local embeddings in: {embeddings_dir}")
            
            if os.path.exists(embeddings_dir):
                total_deleted = 0
                # Walk through all subdirectories
                for root, dirs, files in os.walk(embeddings_dir):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        if os.path.isfile(file_path):
                            try:
                                os.remove(file_path)
                                total_deleted += 1
                                logger.info(f"Deleted: {file_path}")
                            except OSError as e:
                                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
                if total_deleted > 0:
                    logger.info(f"Successfully deleted {total_deleted} embedding files")
                else:
                    logger.info("No embedding files found to delete")
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