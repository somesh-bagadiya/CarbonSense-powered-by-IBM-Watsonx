import os
import logging
from typing import Dict, List, Any
import ibm_boto3
from ibm_botocore.client import Config
from ..config.config_manager import ConfigManager

class COSService:
    """Service for interacting with IBM Cloud Object Storage."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the COS service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.cos_config = config.get_cos_config()
        self.client = self._init_cos_client()
        
    def _init_cos_client(self) -> Any:
        """Initialize IBM Cloud Object Storage client."""
        return ibm_boto3.client(
            "s3",
            ibm_api_key_id=self.cos_config["api_key"],
            ibm_service_instance_id=self.cos_config["instance_id"],
            config=Config(signature_version="oauth"),
            endpoint_url=self.cos_config["endpoint"]
        )
    
    def upload_file(self, file_path: str, object_key: str) -> bool:
        """Upload a file to IBM COS.
        
        Args:
            file_path: Local path to the file
            object_key: Key to store the file as in COS
            
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return False
                
            with open(file_path, "rb") as data:
                self.client.put_object(
                    Bucket=self.cos_config["bucket"],
                    Key=object_key,
                    Body=data
                )
            logging.info(f"Uploaded: {file_path} -> COS as {object_key}")
            return True
            
        except Exception as e:
            logging.error(f"Error uploading {file_path}: {str(e)}")
            return False
    
    def download_file(self, object_key: str, local_path: str) -> bool:
        """Download a file from IBM COS.
        
        Args:
            object_key: Key of the file in COS
            local_path: Local path to save the file
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            self.client.download_file(
                Bucket=self.cos_config["bucket"],
                Key=object_key,
                Filename=local_path
            )
            logging.info(f"Downloaded: {object_key} -> {local_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error downloading {object_key}: {str(e)}")
            return False
    
    def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the COS bucket.
        
        Returns:
            List of dictionaries containing file information
        """
        try:
            objects = self.client.list_objects_v2(Bucket=self.cos_config["bucket"])
            if "Contents" not in objects:
                logging.warning("No files found in the bucket")
                return []
                
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat()
                }
                for obj in objects["Contents"]
            ]
            
        except Exception as e:
            logging.error(f"Error listing files: {str(e)}")
            return []
    
    def delete_file(self, object_key: str) -> bool:
        """Delete a file from IBM COS.
        
        Args:
            object_key: Key of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            self.client.delete_object(
                Bucket=self.cos_config["bucket"],
                Key=object_key
            )
            logging.info(f"Deleted: {object_key}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting {object_key}: {str(e)}")
            return False
    
    def file_exists(self, object_name: str) -> bool:
        """Check if a file exists in COS.
        
        Args:
            object_name: Name of the file to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.client.head_object(
                Bucket=self.config.get_cos_config()["bucket"],
                Key=object_name
            )
            return True
        except Exception as e:
            if "404" in str(e):
                return False
            logging.error(f"Error checking if file {object_name} exists: {str(e)}")
            return False 