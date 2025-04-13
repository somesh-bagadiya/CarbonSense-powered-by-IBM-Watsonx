import logging
import time
import json
import os
from typing import List, Dict, Any, Optional
import litellm
from litellm import completion
from ..config.config_manager import ConfigManager
from ..utils.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

# Configure LiteLLM globally
litellm.set_verbose = False
os.environ["LITELLM_LOG_LEVEL"] = "WARNING"

class LiteLLMWatsonxService:
    """Service for interacting with WatsonX through LiteLLM."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the LiteLLM WatsonX service."""
        self.config = config
        
        # Set up WatsonX configuration
        self.watsonx_config = {
            "model": "watsonx/meta-llama/llama-2-70b-instruct",
            "api_base": os.getenv("WATSONX_URL"),
            "api_key": os.getenv("WATSONX_APIKEY"),
            "api_version": os.getenv("WATSONX_VERSION", "2023-05-29"),
            "project_id": os.getenv("WATSONX_PROJECT_ID")
        }
    
    def _format_error_response(self, exception: Exception) -> Dict[str, Any]:
        """Format error response in a consistent structure."""
        error_info = {
            "error": {
                "message": str(exception),
                "type": type(exception).__name__,
                "param": None,
                "code": None
            }
        }
        
        if hasattr(exception, 'response'):
            error_info["response"] = exception.response
        if hasattr(exception, 'body'):
            error_info["body"] = exception.body
            
        return error_info

    def _process_stream_chunk(self, chunk):
        """Process a single stream chunk quietly."""
        if not chunk or not chunk.choices:
            return None
        
        choice = chunk.choices[0]
        if not hasattr(choice, 'delta') or not choice.delta:
            return None
            
        return choice.delta.content if hasattr(choice.delta, 'content') else None

    def complete(self, 
                prompt: str,
                max_tokens: int = 1000,
                temperature: float = 0.7,
                stop: Optional[list] = None,
                stream: bool = False,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a completion using WatsonX through LiteLLM.
        
        Args:
            prompt: The prompt to generate completion for
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0-1)
            stop: Optional list of stop sequences
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the completion response
        """
        try:
            # Prepare the messages format
            messages = [{"role": "user", "content": prompt}]
            
            # Call LiteLLM completion
            response = completion(
                model=self.watsonx_config["model"],
                messages=messages,
                api_base=self.watsonx_config["api_base"],
                api_key=self.watsonx_config["api_key"],
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=stream,
                **kwargs
            )
            
            if stream:
                return response  # Return stream object directly
            
            # Format the non-streaming response
            return {
                "choices": [{
                    "text": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason
                }],
                "usage": response.usage._asdict() if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            logger.error(f"Error in WatsonX completion: {str(e)}")
            error_info = self._format_error_response(e)
            return {
                "error": error_info["error"]["message"],
                "choices": [{"text": "", "finish_reason": "error"}],
                "usage": {}
            }