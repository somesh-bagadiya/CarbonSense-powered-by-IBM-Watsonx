import logging
import time
import json
from typing import List, Dict, Any
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ..config.config_manager import ConfigManager
from datetime import datetime

# Configure logging to reduce HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ibm_watsonx_ai.wml_resource").setLevel(logging.WARNING)

class WatsonxService:
    """Service for interacting with IBM Watsonx AI models."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the Watsonx service.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.watsonx_config = config.get_watsonx_config()
        self.credentials = self._init_credentials()
        self.embedding_model_30m = self._init_embedding_model("ibm/slate-30m-english-rtrvr-v2")
        self.embedding_model_125m = self._init_embedding_model("ibm/slate-125m-english-rtrvr-v2")
        self.embedding_model_granite = self._init_embedding_model("ibm/granite-embedding-278m-multilingual")
        self.llm_model = self._init_llm_model()
        
        # Rate limiting configuration
        self.last_api_call = 0
        self.min_call_interval = 0.5  # Minimum time between API calls in seconds
        self.batch_size = 5  # Reduced batch size for better rate control
    
    def _init_credentials(self) -> Credentials:
        """Initialize Watsonx credentials."""
        return Credentials(
            url=self.watsonx_config["url"],
            api_key=self.watsonx_config["api_key"]
        )
    
    def _init_embedding_model(self, model_id: str) -> Embeddings:
        """Initialize Watsonx embedding model.
        
        Args:
            model_id: ID of the embedding model to initialize
            
        Returns:
            Initialized embedding model
        """
        try:
            return Embeddings(
                model_id=model_id,
                project_id=self.watsonx_config["project_id"],
                credentials=self.credentials
            )
        except Exception as e:
            logging.error(f"\n❌ Error initializing embedding model {model_id}: {str(e)}\n")
            raise RuntimeError(f"Error initializing embedding model {model_id}: {str(e)}")
    
    def _init_llm_model(self) -> ModelInference:
        """Initialize Watsonx LLM model."""
        return ModelInference(
            model_id="meta-llama/llama-3-3-70b-instruct",
            project_id=self.watsonx_config["project_id"],
            credentials=self.credentials
        )
    
    def _get_model(self, model_type: str) -> Embeddings:
        """Get the initialized embedding model based on the model type."""
        if model_type == "30m":
            return self.embedding_model_30m
        elif model_type == "125m":
            return self.embedding_model_125m
        elif model_type == "granite":
            return self.embedding_model_granite
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be '30m', '125m', or 'granite'")
    
    def _wait_for_rate_limit(self, retry_count: int):
        """Wait if necessary to respect rate limits with exponential backoff."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        self.last_api_call = time.time()
    
    def generate_embeddings(self, texts: List[str], model_type: str = "30m") -> List[List[float]]:
        """Generate embeddings for a list of texts with improved error handling and resource management.
        
        Args:
            texts: List of texts to generate embeddings for
            model_type: Type of model to use ("30m", "125m", or "granite")
            
        Returns:
            List of embeddings
        """
        try:
            model = self._get_model(model_type)
            if not model:
                raise ValueError(f"Model {model_type} not initialized")
            
            all_embeddings = []
            total_texts = len(texts)
            processed_texts = 0
            failed_batches = 0
            max_retries = 3
            batch_size = self._calculate_optimal_batch_size(model_type, total_texts)
            
            logging.info(f"Starting embedding generation for {total_texts} texts using {model_type} model")
            logging.info(f"Using batch size: {batch_size}")
            
            # Initialize performance metrics
            start_time = time.time()
            total_tokens = 0
            
            for i in range(0, total_texts, batch_size):
                batch = texts[i:i + batch_size]
                retry_count = 0
                batch_success = False
                
                while retry_count < max_retries and not batch_success:
                    try:
                        # Monitor memory usage
                        self._check_memory_usage()
                        
                        # Wait for rate limit with exponential backoff
                        self._wait_for_rate_limit(retry_count)
                        
                        # Process batch
                        batch_start = time.time()
                        embeddings = model.embed_documents(batch)
                        batch_time = time.time() - batch_start
                        
                        # Update metrics
                        total_tokens += sum(len(text.split()) for text in batch)
                        processed_texts += len(batch)
                        
                        # Validate embeddings
                        if self._validate_embeddings(embeddings):
                            all_embeddings.extend(embeddings)
                            batch_success = True
                            
                            # Log progress with performance metrics
                            self._log_progress(processed_texts, total_texts, batch_time, total_tokens)
                            
                            # Adjust batch size based on performance
                            batch_size = self._adjust_batch_size(batch_size, batch_time)
                            
                        else:
                            raise ValueError("Invalid embeddings generated")
                            
                    except Exception as e:
                        retry_count += 1
                        failed_batches += 1
                        logging.warning(f"Batch {i//batch_size + 1} failed (attempt {retry_count}/{max_retries}): {str(e)}")
                        
                        if retry_count == max_retries:
                            logging.error(f"Failed to process batch after {max_retries} attempts")
                            # Save failed batch for later processing
                            self._save_failed_batch(batch, model_type)
                            break
                        
                        # Exponential backoff
                        time.sleep(2 ** retry_count)
            
            # Log final statistics
            total_time = time.time() - start_time
            self._log_final_stats(total_texts, processed_texts, failed_batches, total_time, total_tokens)
            
            return all_embeddings
            
        except Exception as e:
            logging.error(f"\n❌ Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Error generating embeddings: {str(e)}")

    def _calculate_optimal_batch_size(self, model_type: str, total_texts: int) -> int:
        """Calculate optimal batch size based on model type and total texts."""
        base_batch_size = {
            "30m": 100,
            "125m": 50,
            "granite": 25
        }.get(model_type, 50)
        
        # Adjust based on total texts
        if total_texts < 100:
            return min(base_batch_size, total_texts)
        elif total_texts < 1000:
            return base_batch_size
        else:
            return min(base_batch_size * 2, 200)

    def _check_memory_usage(self):
        """Check and log memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        if memory_percent > 90:
            logging.warning(f"High memory usage: {memory_percent:.1f}%")
            # Implement memory cleanup if needed
            import gc
            gc.collect()

    def _validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Validate generated embeddings."""
        if not embeddings:
            return False
            
        # Check embedding dimensions
        expected_dim = {
            "30m": 384,
            "125m": 768,
            "granite": 768
        }.get(self.model_type, 768)
        
        return all(len(emb) == expected_dim for emb in embeddings)

    def _adjust_batch_size(self, current_batch_size: int, batch_time: float) -> int:
        """Adjust batch size based on processing time."""
        if batch_time < 1.0:  # Fast processing
            return min(current_batch_size * 2, 200)
        elif batch_time > 5.0:  # Slow processing
            return max(current_batch_size // 2, 10)
        return current_batch_size

    def _log_progress(self, processed: int, total: int, batch_time: float, total_tokens: int):
        """Log detailed progress information."""
        progress = (processed / total) * 100
        tokens_per_second = total_tokens / (time.time() - self.start_time)
        
        logging.info(
            f"⏳ Progress: {processed}/{total} ({progress:.1f}%) | "
            f"Batch time: {batch_time:.2f}s | "
            f"Tokens/s: {tokens_per_second:.1f}"
        )

    def _log_final_stats(self, total_texts: int, processed_texts: int, failed_batches: int, 
                        total_time: float, total_tokens: int):
        """Log final statistics."""
        success_rate = (processed_texts / total_texts) * 100
        tokens_per_second = total_tokens / total_time
        
        logging.info("\n📊 Final Statistics:")
        logging.info(f"Total texts: {total_texts}")
        logging.info(f"Processed texts: {processed_texts}")
        logging.info(f"Failed batches: {failed_batches}")
        logging.info(f"Success rate: {success_rate:.1f}%")
        logging.info(f"Total time: {total_time:.2f}s")
        logging.info(f"Average tokens/second: {tokens_per_second:.1f}")

    def _save_failed_batch(self, batch: List[str], model_type: str):
        """Save failed batch for later processing."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"failed_batch_{model_type}_{timestamp}.json"
            
            data = {
                "model_type": model_type,
                "timestamp": timestamp,
                "texts": batch
            }
            
            with open(filename, "w") as f:
                json.dump(data, f)
                
            logging.info(f"Saved failed batch to {filename}")
        except Exception as e:
            logging.error(f"Failed to save failed batch: {str(e)}")
    
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
        """Generate embedding for a single text chunk.
        
        Args:
            text: Text to generate embedding for
            model_type: Type of model to use ("30m", "125m", or "granite")
            
        Returns:
            List of floats representing the embedding
        """
        try:
            if model_type == "30m":
                model = self.embedding_model_30m
            elif model_type == "125m":
                model = self.embedding_model_125m
            elif model_type == "granite":
                model = self.embedding_model_granite
            else:
                raise ValueError(f"Invalid model type: {model_type}. Must be '30m', '125m', or 'granite'")
            
            # Generate embedding
            response = model.embed_documents([text])
            if response and len(response) > 0:
                return response[0]
            return None
            
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            return None 