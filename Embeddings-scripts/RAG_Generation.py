import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymilvus import connections, Collection
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes

class ConfigManager:
    """Manages configuration and environment variables."""
    
    def __init__(self):
        load_dotenv(override=True)
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validates required environment variables."""
        required_vars = {
            "COS_API_KEY": "IBM Cloud Object Storage API Key",
            "WATSON_STUDIO_PROJECT_ID": "Watson Studio Project ID",
            "MILVUS_GRPC_HOST": "Milvus GRPC Host",
            "MILVUS_GRPC_PORT": "Milvus GRPC Port",
            "MILVUS_CERT_PATH": "Milvus Certificate Path"
        }
        
        missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

class WatsonxManager:
    """Manages Watsonx AI services."""
    
    def __init__(self, api_key: str, project_id: str):
        self.client = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=api_key)
        self.project_id = project_id
        self.embedding_model = self._init_embedding_model()
        self.llm_model = self._init_llm_model()
        
    def _init_embedding_model(self) -> Embeddings:
        """Initializes Watsonx embedding model."""
        return Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            project_id=self.project_id,
            credentials=self.client
        )
        
    def _init_llm_model(self) -> ModelInference:
        """Initializes Watsonx LLM model."""
        return ModelInference(
            model_id=ModelTypes.GRANITE_13B_INSTRUCT_V2,
            project_id=self.project_id,
            credentials=self.client
        )

class MilvusManager:
    """Manages Milvus vector database operations."""
    
    def __init__(self, host: str, port: int, api_key: str, cert_path: str):
        self._connect_to_milvus(host, port, api_key, cert_path)
        
    def _connect_to_milvus(self, host: str, port: int, api_key: str, cert_path: str) -> None:
        """Establishes connection to Milvus database."""
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                user="ibmlhapikey",
                password=api_key,
                secure=True,
                server_ca=cert_path
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")
    
    def search_collection(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches Milvus collection for similar vectors.
        
        Args:
            collection_name: Name of the collection to search
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
            raise RuntimeError(f"Error searching Milvus collection: {str(e)}")

class RAGGenerator:
    """Handles RAG-based question answering."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.watsonx = WatsonxManager(
            os.getenv("COS_API_KEY"),
            os.getenv("WATSON_STUDIO_PROJECT_ID")
        )
        self.milvus = MilvusManager(
            os.getenv("MILVUS_GRPC_HOST"),
            int(os.getenv("MILVUS_GRPC_PORT")),
            os.getenv("COS_API_KEY"),
            os.getenv("MILVUS_CERT_PATH")
        )
    
    def generate_answer(self, query: str, collection_name: str = "carbon_embeddings", top_k: int = 5) -> str:
        """Generates an answer using RAG pipeline.
        
        Args:
            query: User's question
            collection_name: Name of the Milvus collection
            top_k: Number of context chunks to retrieve
            
        Returns:
            Generated answer
        """
        try:
            # Step 1: Embed the query
            query_vector = self.watsonx.embedding_model.embed_documents([query])[0]
            logging.debug("Query embedding generated successfully")
            
            # Step 2: Search Milvus
            results = self.milvus.search_collection(collection_name, query_vector, top_k)
            logging.debug(f"Milvus search completed, found {len(results)} results")
            
            # Step 3: Gather context
            context_chunks = []
            for hit in results:
                chunk = hit.entity.get("chunk_text")
                file_name = hit.entity.get("file_name")
                if chunk and file_name:
                    context_chunks.append(f"From {file_name}:\n{chunk.strip()}")
            
            if not context_chunks:
                logging.warning("No relevant context found in the search results")
                return "I cannot find information about that in the available data."
            
            context = "\n\n".join(context_chunks)
            logging.info(f"Retrieved {len(context_chunks)} relevant chunks from Milvus")
            
            # Step 4: Generate response
            prompt = self._construct_prompt(query, context)
            response = self.watsonx.llm_model.generate(prompt=prompt)
            return response['results'][0]['generated_text']
            
        except Exception as e:
            logging.error(f"Error in generate_answer: {str(e)}", exc_info=True)
            return f"An error occurred while processing your question: {str(e)}"
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """Constructs the prompt for the LLM.
        
        Args:
            query: User's question
            context: Retrieved context from Milvus
            
        Returns:
            Formatted prompt
        """
        return f"""Use the context below to answer the user's question as clearly and accurately as possible.
If the answer cannot be found in the context, say "I cannot find information about that in the available data."

Context:
{context}

Question:
{query}

Answer:"""

def main():
    """Main execution function."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Initialize configuration
        config = ConfigManager()
        
        # Initialize RAG generator
        generator = RAGGenerator(config)
        
        # Interactive loop
        print("💬 CarbonSense | Ask your sustainability question")
        while True:
            try:
                query = input("\n❓ Question (or type 'exit'): ").strip()
                if query.lower() == "exit":
                    break
                
                answer = generator.generate_answer(query)
                print("\n🧠 Answer:\n", answer)
                
            except Exception as e:
                logging.error(f"Error during generation: {e}")
                print("An error occurred. Please try again.")
                
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
