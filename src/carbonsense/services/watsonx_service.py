import logging
from typing import List, Dict, Any
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes, ModelTypes
from ..config.config_manager import ConfigManager

class WatsonxService:
    """Service for interacting with IBM Watsonx AI models."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the Watsonx service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.watsonx_config = config.get_watsonx_config()
        self.client = self._init_client()
        self.embedding_model = self._init_embedding_model()
        self.llm_model = self._init_llm_model()
        
    def _init_client(self) -> Credentials:
        """Initialize Watsonx credentials."""
        return Credentials(
            url="https://us-south.ml.cloud.ibm.com",
            api_key=self.watsonx_config["api_key"]
        )
        
    def _init_embedding_model(self) -> Embeddings:
        """Initialize Watsonx embedding model."""
        return Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            project_id=self.watsonx_config["project_id"],
            credentials=self.client
        )
        
    def _init_llm_model(self) -> ModelInference:
        """Initialize Watsonx LLM model."""
        return ModelInference(
            model_id=ModelTypes.GRANITE_13B_INSTRUCT_V2,
            project_id=self.watsonx_config["project_id"],
            credentials=self.client
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            logging.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {str(e)}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM model.
        
        Args:
            prompt: Input prompt for text generation
            **kwargs: Additional parameters for text generation
            
        Returns:
            Generated text
        """
        try:
            response = self.llm_model.generate(prompt=prompt, **kwargs)
            return response['results'][0]['generated_text']
        except Exception as e:
            raise RuntimeError(f"Error generating text: {str(e)}")
    
    def construct_rag_prompt(self, query: str, context: str) -> str:
        """Construct a prompt for RAG-based question answering.
        
        Args:
            query: User's question
            context: Retrieved context from vector database
            
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