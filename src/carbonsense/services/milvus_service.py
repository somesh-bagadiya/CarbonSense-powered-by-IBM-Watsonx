import logging
from typing import List, Dict, Any
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from ..config.config_manager import ConfigManager

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
                password=self.config.get_cos_config()["api_key"],
                secure=True,
                server_ca=self.milvus_config["cert_path"]
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")
    
    def create_collection(self, collection_name: str, dim: int) -> Collection:
        """Create or retrieve a Milvus collection.
        
        Args:
            collection_name: Name of the collection
            dim: Dimension of the vectors
            
        Returns:
            Milvus Collection object
        """
        try:
            collection = Collection(name=collection_name)
            logging.info(f"Collection '{collection_name}' already exists")
            collection.load()
            return collection
        except Exception:
            logging.info(f"Creating new collection '{collection_name}'")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500)
            ]
            schema = CollectionSchema(fields, description="CarbonSense document embeddings")
            collection = Collection(name=collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
            )
            collection.load()
            return collection
    
    def insert_vectors(self, collection_name: str, entities: List[Dict[str, Any]]) -> None:
        """Insert vectors into a Milvus collection.
        
        Args:
            collection_name: Name of the collection
            entities: List of entities to insert
        """
        try:
            collection = Collection(name=collection_name)
            collection.insert(entities)
            collection.flush()
            logging.info(f"Inserted {len(entities)} vectors into {collection_name}")
        except Exception as e:
            raise RuntimeError(f"Error inserting vectors: {str(e)}")
    
    def search_vectors(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in a Milvus collection.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector to search with
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            collection = Collection(name=collection_name)
            collection.load()
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["chunk_text", "file_name"]
            )
            
            return results[0]
        except Exception as e:
            raise RuntimeError(f"Error searching vectors: {str(e)}")
    
    def drop_collection(self, collection_name: str) -> None:
        """Drop a Milvus collection.
        
        Args:
            collection_name: Name of the collection to drop
        """
        try:
            collection = Collection(name=collection_name)
            collection.drop()
            logging.info(f"Dropped collection: {collection_name}")
        except Exception as e:
            raise RuntimeError(f"Error dropping collection: {str(e)}") 