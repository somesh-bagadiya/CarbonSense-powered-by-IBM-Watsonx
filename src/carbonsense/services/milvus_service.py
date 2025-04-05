import logging
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    IndexType
)
from pymilvus.client.types import MetricType
from ..config.config_manager import ConfigManager
from ..utils.logger import setup_logger
import traceback
import numpy as np
from datetime import datetime
import json

# Set up logger
logger = setup_logger(__name__)

class MilvusService:
    """Service for interacting with Milvus vector database."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the Milvus service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.milvus_config = config.get_milvus_config()
        self._connect_to_milvus()
        
    def _connect_to_milvus(self) -> None:
        """Establish connection to Milvus database."""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_config["host"],
                port=self.milvus_config["port"],
                user="ibmlhapikey",
                password=self.config.get_watsonx_config()["api_key"],
                secure=True,
                server_ca=self.milvus_config["cert_path"]
            )
            logger.info(f"Connected to Milvus at {self.milvus_config['host']}:{self.milvus_config['port']}")
        except Exception as e:
            error_msg = (
                f"Failed to connect to Milvus. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Milvus.
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            exists = utility.has_collection(collection_name)
            logger.info(f"Collection '{collection_name}' {'exists' if exists else 'does not exist'}")
            return exists
        except Exception as e:
            error_msg = (
                f"Error checking collection existence for '{collection_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            return False
    
    def create_collection(self, collection_name: str, dim: int) -> None:
        """Create a new collection in Milvus with enhanced metadata support.
        
        Args:
            collection_name: Name of the collection to create
            dim: Dimension of the vectors to be stored
        """
        try:
            if self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return

            # Define schema with enhanced metadata fields
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="model_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="total_chunks", dtype=DataType.INT64),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="processing_info", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Carbon emissions data embeddings with enhanced metadata"
            )
            
            # Create collection
            collection = Collection(name=collection_name, schema=schema)
            logger.info(f"Created collection '{collection_name}' with dimension {dim}")
            
            # Create indexes for efficient search
            self._create_indexes(collection)
            
        except Exception as e:
            error_msg = (
                f"Error creating collection '{collection_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise
    
    def _create_indexes(self, collection: Collection) -> None:
        """Create necessary indexes for efficient search and filtering."""
        try:
            # Drop existing indexes if any
            try:
                collection.drop_index()
                logger.info("Dropped existing indexes")
            except Exception as e:
                logger.info("No existing indexes to drop")

            # Create vector index for similarity search
            vector_index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=vector_index_params)
            logger.info("Created vector index for similarity search")
            
            # Create indexes for filtering fields
            # For numeric fields (chunk_index)
            collection.create_index(
                field_name="chunk_index",
                index_params={"index_type": "STL_SORT"}
            )
            logger.info("Created index for chunk_index")
            
            # For string fields (source_file, model_type, version)
            string_fields = ["source_file", "model_type", "version"]
            for field in string_fields:
                collection.create_index(
                    field_name=field,
                    index_params={"index_type": "TRIE"}
                )
                logger.info(f"Created index for field: {field}")
            
            # Load the collection after creating indexes
            collection.load()
            logger.info("Collection loaded after index creation")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            raise
    
    def _validate_entity(self, entity: Dict[str, Any]) -> bool:
        """Validate an entity before insertion with enhanced checks."""
        try:
            # Required fields
            required_fields = {
                "id": str,
                "text": str,
                "embedding": list,
                "source_file": str,
                "model_type": str,
                "version": str,
                "chunk_index": int,
                "total_chunks": int
            }
            
            # Check required fields
            for field, field_type in required_fields.items():
                if field not in entity:
                    logger.error(f"Missing required field '{field}' in entity")
                    return False
                
                if not isinstance(entity[field], field_type):
                    logger.error(f"Field '{field}' must be of type {field_type.__name__}")
                    return False
                
            # Validate embedding
            if not isinstance(entity["embedding"], list) or not all(isinstance(x, (int, float)) for x in entity["embedding"]):
                logger.error("Embedding must be a list of numbers")
                return False
        
            return True
            
        except Exception as e:
            logger.error(f"Error validating entity: {str(e)}")
            return False
    
    def insert_vectors(self, collection_name: str, entities: List[Dict[str, Any]]) -> None:
        """Insert vectors into a collection with simplified error handling."""
        try:
            if not entities:
                logger.warning("No entities to insert")
                return

            # Ensure collection exists
            if not self.collection_exists(collection_name):
                if entities and 'embedding' in entities[0]:
                    dim = len(entities[0]['embedding'])
                    logger.info(f"Creating collection {collection_name} with dimension {dim}")
                    self.create_collection(collection_name, dim)
                else:
                    raise ValueError("No embeddings available to determine dimension")

            # Load collection
            collection = Collection(collection_name)
            collection.load()
            
            # Process entities into a simpler format with strict type control
            processed_entities = []
            
            # Get the embedding dimension from first entity
            embedding_dim = len(entities[0]['embedding']) if entities and 'embedding' in entities[0] else 0
            
            for entity in entities:
                # Convert all fields to their proper types
                processed_entity = {
                    "id": str(entity.get("id", "")),
                    "text": str(entity.get("text", "")),
                    "embedding": [float(x) for x in entity.get("embedding", [])[:embedding_dim]],
                    "source_file": str(entity.get("source_file", "")),
                    "model_type": str(entity.get("model_type", "")),
                    "version": str(entity.get("version", "1.0")),
                    "chunk_index": int(entity.get("chunk_index", 0)),
                    "total_chunks": int(entity.get("total_chunks", 0)),
                    "timestamp": int(entity.get("timestamp", int(datetime.now().timestamp()))),
                    "metadata": str(entity.get("metadata", "{}")),
                    "processing_info": str(entity.get("processing_info", "{}"))
                }
                processed_entities.append(processed_entity)
            
            # Insert data directly
            try:
                # Insert as a single entity at a time to avoid type mismatches
                for entity in processed_entities:
                    collection.insert([entity])
            
                # Flush to ensure all data is written
                collection.flush()
                logger.info(f"Successfully inserted {len(processed_entities)} entities")
                
            except Exception as e:
                logger.error(f"Error inserting entities: {str(e)}")
            
        except Exception as e:
            error_msg = (
                f"Error inserting vectors into collection '{collection_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise
    
    def _calculate_optimal_batch_size(self, total_entities: int) -> int:
        """Calculate optimal batch size based on total entities."""
        if total_entities < 100:
            return total_entities
        elif total_entities < 1000:
            return 100
        elif total_entities < 10000:
            return 500
        else:
            return 1000
    
    def query_vectors(self, collection_name: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query vectors from a collection.
        
        Args:
            collection_name: Name of the collection to query
            query: Query string
            limit: Maximum number of results to return
            
        Returns:
            List of query results
        """
        try:
            if not self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return []

            collection = Collection(collection_name)
            collection.load()
            
            results = collection.query(
                expr=query,
                output_fields=["id", "text", "source_file", "model_type", "timestamp"],
                limit=limit
            )
            
            logger.info(f"Retrieved {len(results)} results from collection '{collection_name}'")
            return results
            
        except Exception as e:
            error_msg = (
                f"Error querying vectors from collection '{collection_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            return []
    
    def verify_collection(self, collection_name: str) -> Dict[str, Any]:
        """Verify collection health and statistics."""
        try:
            if not self.collection_exists(collection_name):
                return {"error": f"Collection '{collection_name}' does not exist"}

            collection = Collection(collection_name)
            collection.load()
            
            # Get collection statistics
            stats = {
                "num_entities": collection.num_entities,
                "schema": collection.schema,
                "indexes": collection.indexes,
                "partitions": collection.partitions
            }
            
            # Get unique files and versions
            stats["unique_files"] = self._get_unique_values(collection, "source_file")
            stats["unique_versions"] = self._get_unique_values(collection, "version")
            
            # Get model types used
            stats["model_types"] = self._get_unique_values(collection, "model_type")
            
            # Check for data consistency
            try:
                # Check for valid embeddings by querying a small sample
                results = collection.query(
                    expr="chunk_index >= 0",
                    output_fields=["id", "embedding"],
                    limit=1
                )
                stats["has_valid_embeddings"] = len(results) > 0 and len(results[0]["embedding"]) > 0
                
                # Check for valid metadata
                results = collection.query(
                    expr="metadata != ''",
                    output_fields=["id", "metadata"],
                    limit=1
                )
                stats["has_valid_metadata"] = len(results) > 0 and results[0]["metadata"] != ""
                
            except Exception as e:
                logger.error(f"Error checking data consistency: {str(e)}")
                stats["data_consistency_error"] = str(e)
            
            return stats
            
        except Exception as e:
            return {"error": f"Error verifying collection: {str(e)}"}

    def _get_unique_values(self, collection: Collection, field: str) -> List[str]:
        """Get unique values for a field in the collection."""
        try:
            unique_values = set()
            offset = 0
            batch_size = 1000
            
            while True:
                results = collection.query(
                    expr=f"{field} != ''",
                    output_fields=[field],
                    offset=offset,
                    limit=batch_size
                )
                
                if not results:
                    break
                    
                unique_values.update(result[field] for result in results)
                offset += batch_size
                
                # If we got fewer results than the batch size, we've reached the end
                if len(results) < batch_size:
                    break
            
            return list(unique_values)
        except Exception as e:
            logger.error(f"Error getting unique values for field {field}: {str(e)}")
            return []
    
    def cleanup_collection(self, collection_name: str) -> None:
        """Clean up a collection by dropping it.
        
        Args:
            collection_name: Name of the collection to clean up
        """
        try:
            if not self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' does not exist, nothing to clean up")
                return

            utility.drop_collection(collection_name)
            logger.info(f"Successfully dropped collection '{collection_name}'")
            
        except Exception as e:
            error_msg = (
                f"Error cleaning up collection '{collection_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            raise

    def search_vectors(self, collection_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in a collection.
        
        Args:
            collection_name: Name of the collection to search
            query_embedding: Query vector to search for
            top_k: Number of results to return
            
        Returns:
            List of search results with text and similarity scores
        """
        try:
            if not self.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' does not exist")
                return []

            collection = Collection(collection_name)
            collection.load()
            
            # Prepare search parameters
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            # Search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text", "source_file"]
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    formatted_results.append({
                        "text": hit.entity.get('text'),
                        "source_file": hit.entity.get('source_file'),
                        "score": hit.distance  # Use distance instead of score
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} results from collection '{collection_name}'")
            return formatted_results
            
        except Exception as e:
            error_msg = (
                f"Error searching vectors in collection '{collection_name}'. Details:\n"
                f"- Error Type: {type(e).__name__}\n"
                f"- Error Message: {str(e)}\n"
                f"- Traceback:\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            return [] 