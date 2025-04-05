import os
import sys
import logging
import socket
from pathlib import Path
from src.carbonsense.config.config_manager import ConfigManager
from src.carbonsense.services.embedding_storage_service import EmbeddingStorageService

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('storage_init.log')
        ]
    )

def check_milvus_connection(config):
    """Check if Milvus server is accessible."""
    try:
        milvus_config = config.get_milvus_config()
        host = milvus_config["host"]
        port = int(milvus_config["port"])
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result != 0:
            raise RuntimeError(f"Milvus server is not accessible at {host}:{port}")
        
        logging.info(f"Successfully connected to Milvus server at {host}:{port}")
        
    except Exception as e:
        logging.error(f"Error checking Milvus connection: {str(e)}")
        raise

def check_prerequisites(config):
    """Check if all prerequisites are met."""
    try:
        # Check Milvus connection
        check_milvus_connection(config)
        
        # Check if required directories exist
        base_dir = Path(config.get_storage_config()["base_dir"])
        if base_dir.exists() and not base_dir.is_dir():
            raise RuntimeError(f"Path exists but is not a directory: {base_dir}")
        
        # Check if we have write permissions
        if not os.access(base_dir.parent, os.W_OK):
            raise PermissionError(f"No write permission for directory: {base_dir.parent}")
        
        logging.info("All prerequisites met")
        
    except Exception as e:
        logging.error(f"Prerequisite check failed: {str(e)}")
        raise

def init_storage():
    """Initialize storage structure and Milvus collections."""
    try:
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize storage service
        storage_service = EmbeddingStorageService(config)
        
        # Create storage directories
        base_dir = Path(config.get_storage_config()["base_dir"])
        base_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created base directory: {base_dir}")
        
        # Create model-specific directories
        for model_type in ["30m", "125m", "granite"]:
            model_dir = base_dir / f"embeddings_{model_type}"
            model_dir.mkdir(exist_ok=True)
            logging.info(f"Created directory for {model_type} model: {model_dir}")
        
        logging.info("Storage initialization completed successfully")
        
    except Exception as e:
        logging.error(f"Error initializing storage: {str(e)}")
        raise

def main():
    """Main function to initialize storage."""
    try:
        # Setup logging
        setup_logging()
        
        # Initialize configuration
        config = ConfigManager()
        
        # Check prerequisites
        check_prerequisites(config)
        
        # Initialize storage
        init_storage()
        
        logging.info("Storage initialization completed successfully")
        
    except Exception as e:
        logging.error(f"Storage initialization failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 