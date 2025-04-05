import os
import logging
from typing import Dict
from dotenv import load_dotenv

class ConfigManager:
    """Manages configuration and environment variables for the CarbonSense application."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        load_dotenv(override=True)
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validates required environment variables."""
        required_vars = {
            "COS_API_KEY": "IBM Cloud Object Storage API Key",
            "COS_INSTANCE_ID": "IBM Cloud Object Storage Instance ID",
            "COS_ENDPOINT": "IBM Cloud Object Storage Endpoint",
            "BUCKET_NAME": "IBM Cloud Object Storage Bucket Name",
            "WATSON_STUDIO_PROJECT_ID": "Watson Studio Project ID",
            "MILVUS_GRPC_HOST": "Milvus GRPC Host",
            "MILVUS_GRPC_PORT": "Milvus GRPC Port",
            "MILVUS_CERT_PATH": "Milvus Certificate Path"
        }
        
        missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def get_cos_config(self) -> Dict[str, str]:
        """Get IBM Cloud Object Storage configuration.
        
        Returns:
            Dictionary containing COS configuration
        """
        return {
            "api_key": os.getenv("COS_API_KEY"),
            "instance_id": os.getenv("COS_INSTANCE_ID"),
            "endpoint": os.getenv("COS_ENDPOINT"),
            "bucket_name": os.getenv("BUCKET_NAME")
        }
    
    def get_milvus_config(self) -> Dict[str, str]:
        """Get Milvus configuration.
        
        Returns:
            Dictionary containing Milvus configuration
        """
        return {
            "host": os.getenv("MILVUS_GRPC_HOST"),
            "port": os.getenv("MILVUS_GRPC_PORT"),
            "cert_path": os.getenv("MILVUS_CERT_PATH")
        }
    
    def get_watsonx_config(self) -> Dict[str, str]:
        """Get Watsonx configuration.
        
        Returns:
            Dictionary containing Watsonx configuration
        """
        return {
            "api_key": os.getenv("COS_API_KEY"),
            "project_id": os.getenv("WATSON_STUDIO_PROJECT_ID")
        } 