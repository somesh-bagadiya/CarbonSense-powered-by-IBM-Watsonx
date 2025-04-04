import os
import json
import logging
from io import BytesIO
from typing import List, Dict, Any
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from docx import Document
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

class ConfigManager:
    """Manages configuration and environment variables."""
    
    def __init__(self):
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
            "MILVUS_GRPC_PORT": "Milvus GRPC Port"
        }
        
        missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

class DocumentProcessor:
    """Handles document processing and text extraction."""
    
    @staticmethod
    def extract_text_from_docx(file_bytes: bytes) -> str:
        """Extracts text from a DOCX file.
        
        Args:
            file_bytes: Raw bytes of the DOCX file
            
        Returns:
            Extracted text as a single string
        """
    doc = Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

    @staticmethod
    def chunk_text(text: str, max_chars: int = 400) -> List[str]:
        """Splits text into chunks of specified maximum length.
        
        Args:
            text: Input text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
    lines = text.splitlines()
    chunks = []
    current_chunk = ""
        
    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_chars:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
                
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

class MilvusManager:
    """Manages Milvus vector database operations."""
    
    def __init__(self, host: str, port: str, api_key: str):
        self._connect_to_milvus(host, port, api_key)
        
    def _connect_to_milvus(self, host: str, port: str, api_key: str) -> None:
        """Establishes connection to Milvus database."""
        try:
            connections.connect(
                "default",
                host=host,
                port=port,
                user="ibmlhapikey",
                password=api_key,
                secure=True
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")
    
    def create_collection(self, collection_name: str, dim: int) -> Collection:
        """Creates or retrieves a Milvus collection.
        
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

class EmbeddingGenerator:
    """Handles document embedding generation and storage."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.cos_client = self._init_cos_client()
        self.embedding_model = self._init_embedding_model()
        self.milvus_manager = MilvusManager(
            os.getenv("MILVUS_GRPC_HOST"),
            os.getenv("MILVUS_GRPC_PORT"),
            os.getenv("COS_API_KEY")
        )
        
    def _init_cos_client(self) -> Any:
        """Initializes IBM Cloud Object Storage client."""
        return ibm_boto3.client(
            "s3",
            ibm_api_key_id=os.getenv("COS_API_KEY"),
            ibm_service_instance_id=os.getenv("COS_INSTANCE_ID"),
            config=Config(signature_version="oauth"),
            endpoint_url=os.getenv("COS_ENDPOINT")
        )
        
    def _init_embedding_model(self) -> Embeddings:
        """Initializes Watsonx embedding model."""
        client = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=os.getenv("COS_API_KEY"))
        return Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            project_id=os.getenv("WATSON_STUDIO_PROJECT_ID"),
            credentials=client
        )
    
    def process_file(self, object_name: str) -> None:
        """Processes a single file from COS.
        
        Args:
            object_name: Name of the file in COS
        """
        try:
            file_obj = self.cos_client.get_object(Bucket=os.getenv("BUCKET_NAME"), Key=object_name)
            file_text = self._extract_file_text(file_obj, object_name)
            
            chunks = DocumentProcessor.chunk_text(file_text)
            if len(chunks) > 1:
                logging.info(f"File {object_name} split into {len(chunks)} chunks.")
            
            embeddings = self.embedding_model.embed_documents(chunks)
            logging.info(f"Embeddings generated for {object_name}.")
            
            self._save_embeddings(object_name, chunks, embeddings)
            self._store_in_milvus(object_name, chunks, embeddings)
            
        except Exception as e:
            logging.error(f"Error processing {object_name}: {str(e)}")
            raise
    
    def _extract_file_text(self, file_obj: Any, object_name: str) -> str:
        """Extracts text from file based on its type."""
        lower_name = object_name.lower()
        if lower_name.endswith(".docx"):
            return DocumentProcessor.extract_text_from_docx(file_obj["Body"].read())
        return file_obj["Body"].read().decode("utf-8")
    
    def _save_embeddings(self, object_name: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Saves embeddings locally and to COS."""
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
        self.cos_client.upload_file(Filename=local_file, Bucket=os.getenv("BUCKET_NAME"), Key=cos_key)
        logging.info(f"Uploaded to COS: {cos_key}")

    def _store_in_milvus(self, object_name: str, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Stores embeddings in Milvus vector database."""
        dim = len(embeddings[0])
        collection = self.milvus_manager.create_collection("carbon_embeddings", dim)
        
        entities = [
            {
                "embedding": embedding,
                "chunk_text": chunk,
                "file_name": object_name
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]

        collection.insert(entities)
        collection.flush()
        logging.info(f"Inserted {len(chunks)} vectors into Milvus for {object_name}.")

def main():
    """Main execution function."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize embedding generator
        generator = EmbeddingGenerator(config)
        
        # Process files from COS
        response = generator.cos_client.list_objects_v2(Bucket=os.getenv("BUCKET_NAME"))
        contents = response.get("Contents", [])
        
        if not contents:
            logging.warning("No files found in bucket.")
            return
            
        for obj in contents:
            object_name = obj["Key"]
            if not object_name.endswith(".json"):
                generator.process_file(object_name)

logging.info("All files processed successfully.")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
