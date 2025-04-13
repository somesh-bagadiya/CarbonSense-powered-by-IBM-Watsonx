import os
import json
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading
from ..utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)

class CacheManager:
    """Cache manager for AI agent queries and responses."""
    
    def __init__(self, cache_dir: str = None, cache_ttl: int = 86400):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files (defaults to 'cache' in project root)
            cache_ttl: Time to live for cache entries in seconds (defaults to 24 hours)
        """
        if cache_dir is None:
            # Get project root directory
            project_root = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", ".."
            ))
            cache_dir = os.path.join(project_root, "src", "cache")
        
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "query_cache.json")
        self.cache_ttl = cache_ttl
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Thread lock for safe concurrent access
        self._lock = threading.Lock()
        
        # Initialize cache
        self.cache = self._load_cache()
        
        logger.debug(f"CacheManager initialized with TTL of {cache_ttl} seconds")
    
    def _load_cache(self) -> Dict[str, Any]:
        """
        Load the cache from disk.
        
        Returns:
            Dictionary containing the cache
        """
        if not os.path.exists(self.cache_file):
            return {}
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.debug(f"Cache loaded with {len(cache)} entries")
                return cache
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return {}
    
    def _save_cache(self) -> None:
        """Save the cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _hash_query(self, query: str) -> str:
        """
        Generate a hash for a query.
        
        Args:
            query: The query to hash
            
        Returns:
            MD5 hash of the query as a string
        """
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get a cached result for a query if it exists and is not expired.
        
        Args:
            query: The query to get a result for
            
        Returns:
            Cached result or None if not found or expired
        """
        with self._lock:
            # Generate hash for the query
            query_hash = self._hash_query(query)
            
            # Check if query is in cache
            if query_hash not in self.cache:
                return None
                
            # Get cache entry
            entry = self.cache[query_hash]
            
            # Check if entry is expired
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - timestamp > timedelta(seconds=self.cache_ttl):
                # Remove expired entry
                del self.cache[query_hash]
                self._save_cache()
                logger.debug(f"Cache entry for query '{query[:30]}...' expired")
                return None
                
            logger.debug(f"Cache hit for query '{query[:30]}...'")
            return entry["result"]
    
    def set(self, query: str, result: Dict[str, Any]) -> None:
        """
        Set a cache entry for a query.
        
        Args:
            query: The query to set a result for
            result: The result to cache
        """
        with self._lock:
            # Generate hash for the query
            query_hash = self._hash_query(query)
            
            # Create cache entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "result": result
            }
            
            # Save to cache
            self.cache[query_hash] = entry
            self._save_cache()
            logger.debug(f"Cached result for query '{query[:30]}...'")
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self.cache = {}
            self._save_cache()
            logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            # Count valid and expired entries
            now = datetime.now()
            valid_entries = 0
            expired_entries = 0
            
            for query_hash, entry in self.cache.items():
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if now - timestamp > timedelta(seconds=self.cache_ttl):
                    expired_entries += 1
                else:
                    valid_entries += 1
            
            return {
                "total_entries": len(self.cache),
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
                "cache_file": self.cache_file,
                "cache_ttl_seconds": self.cache_ttl
            }