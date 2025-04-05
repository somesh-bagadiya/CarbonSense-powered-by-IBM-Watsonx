import os
import json
import logging
from typing import List, Dict, Any
from ..config.config_manager import ConfigManager
from ..services.cos_service import COSService
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService
from ..utils.document_processor import DocumentProcessor

class EmbeddingGenerator:
    """Main class for generating and storing document embeddings."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the embedding generator.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.cos = COSService(config)
        self.milvus = MilvusService(config)
        self.watsonx = WatsonxService(config)
        
    def process_file(self, object_name: str) -> None:
        """Process a single file from COS.
        
        Args:
            object_name: Name of the file in COS
        """
        try:
            # Skip JSON files
            if object_name.endswith(".json"):
                logging.info(f"Skipping JSON: {object_name}")
                return
                
            logging.info(f"Processing file: {object_name}")
            
            # Get file from COS
            file_obj = self.cos.client.get_object(
                Bucket=self.config.get_cos_config()["bucket_name"],
                Key=object_name
            )
            
            # Process file content
            file_text = DocumentProcessor.process_file_content(
                file_obj["Body"].read(),
                object_name
            )
            
            # Split into chunks
            chunks = DocumentProcessor.chunk_text(file_text)
            if len(chunks) > 1:
                logging.info(f"File {object_name} split into {len(chunks)} chunks.")
            
            # Generate embeddings
            embeddings = self.watsonx.generate_embeddings(chunks)
            logging.info(f"Embeddings generated for {object_name}.")
            
            # Save embeddings
            self._save_embeddings(object_name, chunks, embeddings)
            
            # Store in Milvus
            self._store_in_milvus(object_name, chunks, embeddings)
            
        except Exception as e:
            logging.error(f"Error processing {object_name}: {str(e)}")
            raise
    
    def _save_embeddings(self, object_name: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Save embeddings locally and to COS.
        
        Args:
            object_name: Name of the file
            chunks: List of text chunks
            embeddings: List of embedding vectors
        """
        try:
            # Prepare embedding data
            embedding_data = {
                "file_name": object_name,
                "chunks": chunks,
                "embeddings": embeddings
            }
            
            # Save locally
            local_file = f"embeddings_{object_name.replace('/', '_')}.json"
            with open(local_file, "w") as f:
                json.dump(embedding_data, f)
            logging.info(f"Saved locally: {local_file}")
            
            # Upload to COS
            cos_key = f"embeddings/{object_name.replace('/', '_')}.json"
            self.cos.upload_file(local_file, cos_key)
            logging.info(f"Uploaded to COS: {cos_key}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving embeddings: {str(e)}")
    
    def _store_in_milvus(self, object_name: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Store embeddings in Milvus vector database.
        
        Args:
            object_name: Name of the file
            chunks: List of text chunks
            embeddings: List of embedding vectors
        """
        try:
            # Create or get collection
            dim = len(embeddings[0])
            collection = self.milvus.create_collection("carbon_embeddings", dim)
            
            # Prepare entities
            entities = [
                {
                    "embedding": embedding,
                    "chunk_text": chunk,
                    "file_name": object_name
                }
                for chunk, embedding in zip(chunks, embeddings)
            ]
            
            # Insert into Milvus
            self.milvus.insert_vectors("carbon_embeddings", entities)
            logging.info(f"Inserted {len(chunks)} vectors into Milvus for {object_name}.")
            
        except Exception as e:
            raise RuntimeError(f"Error storing in Milvus: {str(e)}")
    
    def process_all_files(self) -> None:
        """Process all files in the COS bucket."""
        try:
            # List files in bucket
            response = self.cos.client.list_objects_v2(
                Bucket=self.config.get_cos_config()["bucket_name"]
            )
            contents = response.get("Contents", [])
            
            if not contents:
                logging.warning("No files found in bucket.")
                return
                
            # Process each file
            for obj in contents:
                self.process_file(obj["Key"])
                
            logging.info("All files processed successfully.")
            
        except Exception as e:
            raise RuntimeError(f"Error processing files: {str(e)}") 