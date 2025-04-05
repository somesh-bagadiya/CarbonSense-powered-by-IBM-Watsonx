import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    IndexType,
    MetricType
)
from ..services.watsonx_service import WatsonxService

class EmbeddingStorageService:
    """Service for managing embedding storage and versioning using Milvus."""
    
    def __init__(self, config: Any):
        """Initialize the embedding storage service.
        
        Args:
            config: Configuration object containing storage settings
        """
        self.config = config
        self.storage_config = config.get_storage_config()
        self.milvus_config = config.get_milvus_config()
        self.watsonx_service = WatsonxService(config)
        
        # Initialize storage directories
        self._init_storage_dirs()
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        # Initialize Milvus collections
        self._init_milvus_collections()
    
    def _connect_to_milvus(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_config["host"],
                port=self.milvus_config["port"]
            )
            logging.info("Connected to Milvus server")
        except Exception as e:
            logging.error(f"Error connecting to Milvus: {str(e)}")
            raise
    
    def _init_storage_dirs(self):
        """Initialize storage directories for different embedding models."""
        try:
            base_dir = Path(self.storage_config["base_dir"])
            
            # Create base directory if it doesn't exist
            if not base_dir.exists():
                base_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created base directory: {base_dir}")
            
            # Create directories for each model type
            self.model_dirs = {
                "30m": base_dir / "embeddings_30m",
                "125m": base_dir / "embeddings_125m",
                "granite": base_dir / "embeddings_granite"
            }
            
            for model_type, dir_path in self.model_dirs.items():
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Created directory for {model_type} model: {dir_path}")
                
                # Verify directory permissions
                if not os.access(dir_path, os.W_OK):
                    raise PermissionError(f"No write permission for directory: {dir_path}")
            
        except Exception as e:
            logging.error(f"Error initializing storage directories: {str(e)}")
            raise RuntimeError(f"Error initializing storage directories: {str(e)}")
    
    def _init_milvus_collections(self):
        """Initialize Milvus collections for each model type."""
        self.collections = {}
        
        # Model configurations
        model_configs = {
            "30m": {
                "name": "embeddings_30m",
                "dim": 384,  # Slate 30m English RTRVR v2
                "description": "Slate 30m English RTRVR v2 embeddings"
            },
            "125m": {
                "name": "embeddings_125m",
                "dim": 768,  # Slate 125m English RTRVR v2
                "description": "Slate 125m English RTRVR v2 embeddings"
            },
            "granite": {
                "name": "embeddings_granite",
                "dim": 768,  # Granite Embedding 278m Multilingual
                "description": "Granite Embedding 278m Multilingual embeddings"
            }
        }
        
        for model_type, config in model_configs.items():
            collection_name = config["name"]
            
            # Check if collection exists
            if utility.has_collection(collection_name):
                # Load existing collection
                self.collections[model_type] = Collection(collection_name)
                logging.info(f"Loaded existing collection: {collection_name}")
            else:
                # Create new collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=config["dim"]),
                    FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="timestamp", dtype=DataType.INT64),
                    FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="model_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
                ]
                
                schema = CollectionSchema(
                    fields=fields,
                    description=config["description"]
                )
                
                # Create collection
                self.collections[model_type] = Collection(
                    name=collection_name,
                    schema=schema
                )
                
                # Create index for efficient similarity search
                index_params = {
                    "metric_type": MetricType.L2,
                    "index_type": IndexType.IVF_FLAT,
                    "params": {"nlist": 1024}
                }
                self.collections[model_type].create_index(
                    field_name="embedding",
                    index_params=index_params
                )
                
                # Create index for version field to enable filtering
                self.collections[model_type].create_index(
                    field_name="version",
                    index_params={"index_type": "STL_SORT"}
                )
                
                logging.info(f"Created new collection: {collection_name}")
            
            # Load collection into memory
            self.collections[model_type].load()
            logging.info(f"Loaded collection into memory: {collection_name}")
    
    def store_embeddings(self, data: Dict[str, Any], model_type: str, version: str = None, 
                        source_file: str = None, metadata: Dict = None) -> str:
        """Store embeddings in both Milvus and file system.
        
        Args:
            data: Dictionary containing chunks and embeddings
            model_type: Type of model used ("30m", "125m", or "granite")
            version: Optional version string, defaults to timestamp
            source_file: Optional source file name
            metadata: Optional additional metadata
            
        Returns:
            Version string used for storage
        """
        try:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Store in Milvus
            self._store_in_milvus(data, model_type, version, source_file, metadata)
            
            # Store in file system as backup
            file_path = self._store_in_filesystem(data, model_type, version)
            
            logging.info(f"Stored embeddings for model {model_type} with version {version}")
            return version
            
        except Exception as e:
            logging.error(f"Error storing embeddings: {str(e)}")
            raise
    
    def _store_in_milvus(self, data: Dict[str, Any], model_type: str, version: str, 
                        source_file: str = None, metadata: Dict = None):
        """Store embeddings in Milvus."""
        try:
            collection = self.collections[model_type]
            timestamp = int(datetime.now().timestamp())
            
            # Prepare data for insertion
            entities = []
            for i, (text, embedding) in enumerate(zip(data["chunks"], data["embeddings"])):
                entity = {
                    "text": text,
                    "embedding": embedding.tolist(),
                    "version": version,
                    "timestamp": timestamp,
                    "model_type": model_type,
                    "metadata": json.dumps(metadata or {})
                }
                
                if source_file:
                    entity["source_file"] = source_file
                
                entities.append(entity)
            
            # Insert data
            collection.insert(entities)
            collection.flush()
            
            logging.info(f"Stored {len(entities)} embeddings in Milvus for version {version}")
            
        except Exception as e:
            logging.error(f"Error storing embeddings in Milvus: {str(e)}")
            raise
    
    def _store_in_filesystem(self, data: Dict[str, Any], model_type: str, version: str) -> Path:
        """Store embeddings in file system."""
        try:
            dir_path = self.model_dirs[model_type]
            file_path = dir_path / f"embeddings_{version}.json"
            
            # Verify directory exists and is writable
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory does not exist: {dir_path}")
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permission for directory: {dir_path}")
            
            # Convert numpy arrays to lists for JSON serialization
            data_to_store = {
                "chunks": data["chunks"],
                "embeddings": [embedding.tolist() for embedding in data["embeddings"]],
                "version": version,
                "timestamp": int(datetime.now().timestamp()),
                "model_type": model_type
            }
            
            # Write to temporary file first
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data_to_store, f, indent=2)
            
            # Move temporary file to final location
            temp_path.replace(file_path)
            
            logging.info(f"Stored embeddings in file: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error storing embeddings in filesystem: {str(e)}")
            raise RuntimeError(f"Error storing embeddings in filesystem: {str(e)}")
    
    def get_embeddings(self, model_type: str, version: str = None) -> Dict[str, Any]:
        """Retrieve embeddings from storage.
        
        Args:
            model_type: Type of model used ("30m", "125m", or "granite")
            version: Optional version string, defaults to latest
            
        Returns:
            Dictionary containing chunks and embeddings
        """
        try:
            if version is None:
                version = self._get_latest_version(model_type)
            
            # Get from file system
            return self._get_from_filesystem(model_type, version)
            
        except Exception as e:
            logging.error(f"Error retrieving embeddings: {str(e)}")
            raise RuntimeError(f"Error retrieving embeddings: {str(e)}")
    
    def _get_latest_version(self, model_type: str) -> str:
        """Get the latest version string for a model type."""
        dir_path = self.model_dirs[model_type]
        versions = [f.stem.split('_')[1] for f in dir_path.glob('embeddings_*.json')]
        return max(versions) if versions else None
    
    def _get_from_filesystem(self, model_type: str, version: str) -> Dict[str, Any]:
        """Retrieve embeddings from file system."""
        try:
            dir_path = self.model_dirs[model_type]
            file_path = dir_path / f"embeddings_{version}.json"
            
            # Verify file exists and is readable
            if not file_path.exists():
                raise FileNotFoundError(f"Embeddings file not found: {file_path}")
            if not os.access(file_path, os.R_OK):
                raise PermissionError(f"No read permission for file: {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Verify data structure
            required_fields = ["chunks", "embeddings", "version", "timestamp", "model_type"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field in embeddings file: {field}")
            
            # Convert lists back to numpy arrays
            data["embeddings"] = [np.array(embedding) for embedding in data["embeddings"]]
            
            logging.info(f"Retrieved embeddings from file: {file_path}")
            return data
            
        except Exception as e:
            logging.error(f"Error retrieving embeddings from filesystem: {str(e)}")
            raise RuntimeError(f"Error retrieving embeddings from filesystem: {str(e)}")
    
    def search_embeddings(self, query_text: str, model_type: str, 
                         top_k: int = 5, version: str = None) -> List[Dict[str, Any]]:
        """Search for similar embeddings using Milvus.
        
        Args:
            query_text: Text to search for
            model_type: Type of model used ("30m", "125m", or "granite")
            top_k: Number of results to return
            version: Optional version string to filter by
            
        Returns:
            List of search results with text and similarity scores
        """
        try:
            # Generate embedding for query text
            query_embedding = self.watsonx_service.generate_embedding(query_text, model_type)
            
            # Get collection
            collection = self.collections[model_type]
            
            # Prepare search parameters
            search_params = {
                "metric_type": MetricType.L2,
                "params": {"nprobe": 10}
            }
            
            # Prepare expression for version filtering
            expr = None
            if version:
                expr = f'version == "{version}"'
            
            # Search
            results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "version", "source_file", "metadata"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get('text'),
                        "version": hit.entity.get('version'),
                        "source_file": hit.entity.get('source_file'),
                        "score": hit.score,
                        "metadata": json.loads(hit.entity.get('metadata', '{}'))
                    })
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error searching embeddings: {str(e)}")
            raise
    
    def get_embeddings_by_version(self, model_type: str, version: str) -> List[Dict[str, Any]]:
        """Retrieve all embeddings for a specific version.
        
        Args:
            model_type: Type of model used ("30m", "125m", or "granite")
            version: Version string
            
        Returns:
            List of embeddings with metadata
        """
        try:
            collection = self.collections[model_type]
            
            # Query embeddings for specific version
            results = collection.query(
                expr=f'version == "{version}"',
                output_fields=["text", "embedding", "source_file", "metadata"]
            )
            
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving embeddings by version: {str(e)}")
            raise
    
    def get_available_versions(self, model_type: str) -> List[str]:
        """Get list of available versions for a model type.
        
        Args:
            model_type: Type of model used ("30m", "125m", or "granite")
            
        Returns:
            List of version strings
        """
        try:
            collection = self.collections[model_type]
            
            # Query distinct versions
            results = collection.query(
                expr="version != ''",
                output_fields=["version"],
                consistency_level="Strong"
            )
            
            # Extract unique versions
            versions = list(set(result["version"] for result in results))
            versions.sort(reverse=True)  # Sort by most recent first
            
            return versions
            
        except Exception as e:
            logging.error(f"Error retrieving available versions: {str(e)}")
            raise 