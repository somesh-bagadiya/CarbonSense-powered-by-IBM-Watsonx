import os
import json
from typing import Dict, Any
from pathlib import Path
import logging
from dotenv import load_dotenv

class ConfigManager:
    """Manager for application configuration."""
    
    def __init__(self, config_path: str = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config.json')
        self.config = self._load_config()
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
            "MILVUS_CERT_PATH": "Milvus Certificate Path",
            "WATSON_DISCOVERY_API_KEY": "Watson Discovery API Key",
            "WATSON_DISCOVERY_URL": "Watson Discovery Service URL",
            "WATSON_DISCOVERY_PROJECT_ID": "Watson Discovery Project ID"
        }
        
        missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "watsonx": {
                "url": os.getenv('WATSONX_URL', ''),
                "api_key": os.getenv('WATSONX_API_KEY', ''),
                "project_id": os.getenv('WATSON_STUDIO_PROJECT_ID', '')
            },
            "milvus": {
                "host": os.getenv('MILVUS_GRPC_HOST', 'localhost'),
                "port": os.getenv('MILVUS_GRPC_PORT', '19530'),
                "rest_host": os.getenv('MILVUS_REST_HOST', ''),
                "rest_port": os.getenv('MILVUS_REST_PORT', ''),
                "cert_path": os.getenv('MILVUS_CERT_PATH', '')
            },
            "cos": {
                "api_key": os.getenv('COS_API_KEY', ''),
                "instance_id": os.getenv('COS_INSTANCE_ID', ''),
                "endpoint": os.getenv('COS_ENDPOINT', ''),
                "bucket": os.getenv('BUCKET_NAME', '')
            },
            "storage": {
                "base_dir": os.getenv('STORAGE_BASE_DIR', 'embeddings'),
                "models": {
                    "30m": {
                        "dimension": 384,
                        "collection_name": "embeddings_30m"
                    },
                    "125m": {
                        "dimension": 768,
                        "collection_name": "embeddings_125m"
                    },
                    "granite": {
                        "dimension": 768,
                        "collection_name": "embeddings_granite"
                    }
                }
            },
            "chunking": {
                "chunk_size": int(os.getenv('CHUNK_SIZE', '1000')),
                "overlap": int(os.getenv('CHUNK_OVERLAP', '200'))
            }
        }
    
    def get_watsonx_config(self) -> Dict[str, str]:
        """Get Watsonx configuration."""
        return self.config["watsonx"]
    
    def get_milvus_config(self) -> Dict[str, Any]:
        """Get Milvus configuration."""
        return self.config["milvus"]
    
    def get_cos_config(self) -> Dict[str, str]:
        """Get Cloud Object Storage configuration."""
        return self.config["cos"]
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return self.config["storage"]
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type.
        
        Args:
            model_type: Type of model ("30m", "125m", or "granite")
            
        Returns:
            Model configuration
        """
        return self.config["storage"]["models"][model_type]
    
    def get_chunking_config(self) -> Dict[str, int]:
        """Get text chunking configuration.
        
        Returns:
            Dictionary containing chunk_size and overlap settings
        """
        return self.config.get("chunking", {
            "chunk_size": 1000,
            "overlap": 200
        })
    
    def get_web_search_config(self) -> Dict[str, str]:
        """Get web search configuration.
        
        Returns:
            Dictionary containing web search API settings
        """
        return {
            "api_key": os.getenv("GOOGLE_SEARCH_API_KEY", ""),
            "engine_id": os.getenv("GOOGLE_SEARCH_ENGINE_ID", "")
        }
    
    def get_discovery_config(self) -> Dict[str, str]:
        """Get Watson Discovery configuration.
        
        Returns:
            Dictionary containing Watson Discovery configuration
        """
        return {
            "api_key": os.getenv("WATSON_DISCOVERY_API_KEY"),
            "url": os.getenv("WATSON_DISCOVERY_URL"),
            "version": os.getenv("WATSON_DISCOVERY_VERSION", "2023-11-17"),
            "project_id": os.getenv("WATSON_DISCOVERY_PROJECT_ID")
        }
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2) 