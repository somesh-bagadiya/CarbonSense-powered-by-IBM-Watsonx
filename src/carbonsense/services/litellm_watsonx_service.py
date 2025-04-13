import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
import litellm
from litellm import completion
from ..config.config_manager import ConfigManager
from datetime import datetime

# Configure logging to reduce HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)

class LiteLLMWatsonxService:
    """Service for interacting with IBM Watsonx AI models using LiteLLM specifically for CrewAI."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the LiteLLM Watsonx service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.watsonx_config = config.get_watsonx_config()
        
        # Set up environment variables for LiteLLM
        os.environ["WATSONX_URL"] = self.watsonx_config["url"]
        os.environ["WATSONX_APIKEY"] = self.watsonx_config["api_key"]
        os.environ["WATSONX_PROJECT_ID"] = self.watsonx_config["project_id"]
        
        # Model mapping - follows LiteLLM's model naming convention
        self.model_mapping = {
            "chat": "watsonx/meta-llama/llama-3-1-8b-instruct",  # Chat endpoint
            "generation": "watsonx/ibm/granite-13b-chat-v2",     # Generation endpoint
            "embeddings": "watsonx/embedding-models"
        }
        
        # Rate limiting configuration
        self.last_api_call = 0
        self.min_call_interval = 0.5  # Minimum time between API calls in seconds
        
        logging.info("LiteLLM Watsonx service initialized successfully")
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        self.last_api_call = time.time()
    
    def generate_text(self, prompt: str, use_chat: bool = True, **kwargs) -> str:
        """Generate text using the LLM model.
        
        Args:
            prompt: Input prompt for text generation
            use_chat: Whether to use the chat endpoint (True) or generation endpoint (False)
            **kwargs: Additional parameters for text generation
            
        Returns:
            Generated text
        """
        try:
            self._wait_for_rate_limit()
            
            # Prepare messages in the format expected by LiteLLM
            messages = [{"role": "user", "content": prompt}]
            
            # Select model based on endpoint type
            model = self.model_mapping["chat"] if use_chat else self.model_mapping["generation"]
            
            # Call LiteLLM completion
            response = completion(
                model=model,
                messages=messages,
                **kwargs
            )
            
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content
                
            return "No response generated from the model."
            
        except Exception as e:
            logging.error(f"Error generating text with LiteLLM: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_text_for_crew(self, prompt: str, **kwargs) -> str:
        """Generate text using the LLM model specifically for CrewAI integration.
        
        Args:
            prompt: Input prompt for text generation
            **kwargs: Additional parameters for text generation
            
        Returns:
            Generated text
        """
        try:
            # Check if the prompt is valid
            if not isinstance(prompt, str):
                prompt = str(prompt)
            
            # Define generation parameters optimized for CrewAI
            params = {
                "max_tokens": 1000,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
            
            # Update params with any provided kwargs
            params.update(kwargs)
            
            return self.generate_text(prompt, use_chat=True, **params)
            
        except Exception as e:
            logging.error(f"Error in text generation for CrewAI with LiteLLM: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def construct_rag_prompt(self, query: str, context: str) -> str:
        """Construct a prompt for RAG-based question answering.
        
        Args:
            query: User's question
            context: Retrieved context from vector database
            
        Returns:
            Formatted prompt
        """
        return f"""You are an expert in environmental science and carbon emissions. Use the context below to provide a detailed and informative answer to the user's question. Include specific numbers and units when available. If the information cannot be found in the context, say "I cannot find detailed information about that in the available data."

Context:
{context}

Question:
{query}

Provide a detailed answer with the following structure:
1. Direct answer to the question with specific numbers/metrics
2. Additional context or explanation
3. Source information

Answer:"""

    def generate_embedding(self, text: str, model_type: str = "30m") -> List[float]:
        """Generate embedding for a single text chunk using LiteLLM.
        
        Args:
            text: Text to generate embedding for
            model_type: Type of model to use ("30m", "125m", or "granite")
            
        Returns:
            List of floats representing the embedding
        """
        try:
            self._wait_for_rate_limit()
            
            # Map model type to actual model name
            model_names = {
                "30m": "watsonx/ibm/slate-30m-english-rtrvr-v2",
                "125m": "watsonx/ibm/slate-125m-english-rtrvr-v2",
                "granite": "watsonx/ibm/granite-embedding-278m-multilingual"
            }
            
            model = model_names.get(model_type)
            if not model:
                raise ValueError(f"Invalid model type: {model_type}. Must be '30m', '125m', or 'granite'")
            
            # Generate embedding using LiteLLM
            response = litellm.embedding(
                model=model,
                input=[text]  # LiteLLM expects a list of texts
            )
            
            if response and hasattr(response, 'data') and len(response.data) > 0:
                return response.data[0].embedding
            
            raise ValueError("No embedding generated from the model")
            
        except Exception as e:
            logging.error(f"Error generating embedding with LiteLLM: {str(e)}")
            return None