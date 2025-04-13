import logging
from typing import Dict, List, Any, Optional, Tuple
from ..services.milvus_service import MilvusService
from ..services.watsonx_service import WatsonxService
from ..services.discovery_service import DiscoveryService
from ..utils.logger import setup_logger
import requests
from datetime import datetime
import re

# Set up logger with colored output
logger = setup_logger(__name__)

class CarbonAgent:
    """Agent for processing carbon footprint queries with fallback to web search."""
    
    def __init__(self, config: Any):
        """Initialize the carbon agent.
        
        Args:
            config: Configuration object
        """
        logger.info("Initializing CarbonAgent...")
        self.config = config
        self.milvus = MilvusService(config)
        self.watsonx = WatsonxService(config)
        self.discovery = DiscoveryService(config)
        self.confidence_threshold = 0.6  # Adjustable threshold for Milvus search confidence
        self.web_search_api_key = config.get_web_search_config()["api_key"]
        self.web_search_engine_id = config.get_web_search_config()["engine_id"]
        logger.info("✅ CarbonAgent initialized successfully")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query about carbon footprint.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary containing the response and metadata
        """
        logger.info(f"\n🔍 Processing query: {query}")
        try:
            # Step 1: Extract product and quantity
            logger.info("📝 Extracting product and quantity from query...")
            product, quantity = self._extract_product_and_quantity(query)
            
            if not product:
                logger.warning("❌ Could not identify a product in the query")
                return {
                    "error": "Could not identify a product in the query",
                    "suggestion": "Please provide a specific product name in your query."
                }
            
            # Step 2: Search Milvus
            logger.info(f"🔎 Searching Milvus for information about {product}...")
            milvus_results = self._search_milvus(product)
            
            # Step 3: Check confidence and decide on web search
            web_search_used = False
            if milvus_results["confidence"] < self.confidence_threshold:
                logger.info(f"⚠️ Low confidence ({milvus_results['confidence']:.2f}) from Milvus, falling back to web search")
                web_context = self._search_web(product)
                context = self._combine_contexts(milvus_results["context"], web_context)
                web_search_used = True
            else:
                logger.info("✅ Using only Milvus results due to high confidence")
                context = milvus_results["context"]
            
            # Step 4: Generate response
            response = self._generate_response(product, quantity, context)
            logger.info("✅ Response generated successfully")
            
            return {
                "response": response,
                "product": product,
                "quantity": quantity,
                "confidence": milvus_results["confidence"],
                "sources": milvus_results["sources"],
                "web_search_used": web_search_used
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}", exc_info=True)
            return {
                "error": "An unexpected error occurred while processing your query.",
                "details": str(e)
            }
    
    def _extract_product_and_quantity(self, query: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract product and quantity from a query.
        
        Args:
            query: User's query string
            
        Returns:
            Tuple of (product, quantity) or (None, None) if extraction fails
        """
        logger.info("📝 Starting product and quantity extraction...")
        try:
            # Use Watsonx to extract product and quantity
            prompt = f"""
            Extract the product and quantity from this query about carbon footprint.
            The query is: {query}
            
            Strictly return ONLY the product name and quantity in this format:
            Product, Quantity
            Example:
            Banana, 6
            
            If no specific quantity is mentioned, use 1 as default.
            """
            
            params = {
                "MAX_NEW_TOKENS": 50,  # Maximum length of generated text
            }
            
            response = self.watsonx.generate_text(
                prompt=prompt,
                params=params,
            )

            response = self.watsonx.generate_text(prompt)
            logger.info(f"📝 Extraction response: {response}")
            
            # Clean and normalize the response
            response = response.lower().strip()
            
            # Try to extract product and quantity using various patterns
            product = None
            quantity = 1  # Default quantity
            
            # Pattern 1: Look for common quantity indicators
            quantity_patterns = [
                r'(\d+)\s*(kg|kgs|kilogram|kilograms|g|grams|ton|tons|unit|units|piece|pieces)',
                r'(\d+)\s*(of|for)',
                r'(\d+)\s*$'
            ]
            
            for pattern in quantity_patterns:
                match = re.search(pattern, response)
                if match:
                    try:
                        quantity = int(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
            
            # Pattern 2: Look for product mentions
            # Remove quantity-related words to isolate the product
            quantity_words = ['kg', 'kgs', 'kilogram', 'kilograms', 'g', 'grams', 
                            'ton', 'tons', 'unit', 'units', 'piece', 'pieces', 
                            'of', 'for', 'quantity', 'amount']
            product_text = ' '.join(word for word in response.split() 
                                  if word not in quantity_words and not word.isdigit())
            
            # Clean up the product text
            product = product_text.strip()
            if product:
                # Remove any remaining special characters
                product = re.sub(r'[^\w\s]', '', product)
                # Remove extra whitespace
                product = ' '.join(product.split())
            
            if not product:
                logger.warning("❌ Failed to extract product name from response")
                return None, None
                
            logger.info(f"✅ Extracted - Product: {product}, Quantity: {quantity}")
            return product, quantity
            
        except Exception as e:
            logger.error(f"❌ Error extracting product and quantity: {str(e)}", exc_info=True)
            return None, None
    
    def _search_milvus(self, product: str) -> Dict[str, Any]:
        """Search Milvus for carbon footprint information.
        
        Args:
            product: Product name to search for
            
        Returns:
            Dictionary containing context, confidence score, and sources
        """
        logger.info(f"🔎 Starting Milvus search for {product}...")
        try:
            # Generate query embedding using the granite model (3072 dimensions)
            query_embedding = self.watsonx.generate_embedding(
                f"carbon footprint of {product}",
                model_type="granite"
            )
            logger.info("✅ Query embedding generated")
            
            if not query_embedding:
                logger.error("❌ Failed to generate query embedding")
                return {
                    "context": "",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Search in Milvus
            results = self.milvus.search_vectors(
                collection_name="carbon_embeddings_granite",
                query_embedding=query_embedding,
                top_k=5
            )
            logger.info(f"✅ Found {len(results)} results in Milvus")
            
# Print Milvus search results
            print("\n📋 Milvus Search Results:")
            print("=" * 80)
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Score: {result['score']:.4f}")
                print(f"Source: {result['source_file']}")
                print(f"Text: {result['text']}")
            
            if not results:
                logger.warning("⚠️ No results found in Milvus")
                return {
                    "context": "",
                    "confidence": 0.0,
                    "sources": []
                }
            
            # Calculate average confidence score
            confidence = sum(result["score"] for result in results) / len(results)
            logger.info(f"📊 Average confidence score: {confidence:.2f}")
            
            # Extract context and sources
            context = "\n".join(result["text"] for result in results)
            sources = [{"file": result["source_file"], "score": result["score"]} for result in results]
            
            return {
                "context": context,
                "confidence": confidence,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"❌ Error searching Milvus: {str(e)}", exc_info=True)
            return {
                "context": "",
                "confidence": 0.0,
                "sources": []
            }
    
    def _search_web(self, product: str) -> str:
        """Search the web for carbon footprint information using Watson Discovery.
        
        Args:
            product: Product name to search for
            
        Returns:
            Combined context from web search results
        """
        logger.info(f"🌐 Starting web search for {product}...")
        try:
            search_query = f"carbon footprint of {product}"
            results = self.discovery.search_web(search_query, max_results=5)
            logger.info(f"✅ Found {len(results)} web search results")
            
            if not results:
                logger.warning("⚠️ No web search results found")
                return ""
            
            # Extract and format context from results
            context_parts = []

            for result in results:
                # Get the first passage if available, otherwise use the full text
                text = result.get('document_passages', [{}])[0].get('passage_text', result.get('text', ''))
                if text:  # Only include results with content
                    context_parts.append(
                        f"[Source: {result.get('title', 'Unknown')}]\n"
                        f"{text}"
                    )
            
            context = "\n\n".join(context_parts)
            logger.info(f"📝 Web search context length: {len(context)} characters")
            return context
            
        except Exception as e:
            logger.error(f"❌ Error searching web: {str(e)}", exc_info=True)
            return ""
    
    def _combine_contexts(self, milvus_context: str, web_context: str) -> str:
        """Combine contexts from Milvus and web search.
        
        Args:
            milvus_context: Context from Milvus search
            web_context: Context from web search
            
        Returns:
            Combined context string
        """
        if not milvus_context and not web_context:
            return "No relevant information found."
            
        combined = []
        if milvus_context:
            combined.append("From our database:\n" + milvus_context)
        if web_context:
            combined.append("From web sources:\n" + web_context)
            
        return "\n\n".join(combined)
    
    def _generate_response(self, product: str, quantity: int, context: str) -> str:
        """Generate a response using the gathered context.
        
        Args:
            product: Product name
            quantity: Product quantity
            context: Combined context from Milvus and web search
            
        Returns:
            Generated response
        """
        try:
            prompt = f"""
            Based on the following information about the carbon footprint of {product},
            provide a detailed response about the environmental impact of consuming {quantity} {product}(s).
            
            Context:
            {context}
            
            Please include:
            1. The carbon footprint per unit
            2. The total carbon footprint for the specified quantity
            3. Any relevant environmental considerations
            4. Sources of the information
            
            If the information is not sufficient, please indicate what additional data would be needed.
            
            Keep your response concise and focused on the key information.
            """
            
            # Define text generation parameters
            params = {
                "MAX_NEW_TOKENS": 2000,  # Maximum length of generated text
                "MIN_NEW_TOKENS": 100,  # Minimum length to ensure complete answers
                "TEMPERATURE": 0.7,     # Controls randomness (0.0 to 1.0)
                "TOP_P": 0.9,          # Nucleus sampling parameter
                "TOP_K": 50,           # Number of highest probability tokens to consider
                "DECODING_METHOD": "greedy"  # Use greedy decoding for more focused responses
            }
            
            response = self.watsonx.generate_text(
                prompt=prompt,
                params=params
            )
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error while generating the response. Please try again later."