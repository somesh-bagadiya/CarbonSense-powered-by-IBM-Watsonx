import os, json, hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class CacheService:
    """Simple file-based cache service that doesn't depend on Redis."""
    
    def __init__(self, ttl:int = None):
        """Initialize the cache service.
        
        Args:
            ttl: Time to live for cache entries in seconds (defaults to 1 day)
        """
        # Define cache directory in the current working directory
        self.cache_dir = os.path.join(os.getcwd(), "file_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set TTL (default to 1 day)
        self.ttl = ttl or int(os.getenv("CACHE_TTL", "86400"))
        print(f"✅ CacheService initialized with file-based cache in {self.cache_dir}")
        
    def _hash_key(self, key: str) -> str:
        """Create a hash for a key.
        
        Args:
            key: The key to hash
            
        Returns:
            Hashed key safe for filenames
        """
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _get_cache_file(self, key: str) -> str:
        """Get the cache file path for a key.
        
        Args:
            key: The key to get the cache file for
            
        Returns:
            Path to the cache file
        """
        hashed_key = self._hash_key(key)
        return os.path.join(self.cache_dir, f"{hashed_key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.
        
        Args:
            key: The key to get
            
        Returns:
            The cached value or None if not found or expired
        """
        cache_file = self._get_cache_file(key)
        
        # Check if cache file exists
        if not os.path.exists(cache_file):
            return None
            
        try:
            # Read cache file
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                
            # Check if entry is expired
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - timestamp > timedelta(seconds=self.ttl):
                # Remove expired entry
                os.remove(cache_file)
                print(f"Cache entry for '{key[:30]}...' expired")
                return None
                
            print(f"Cache hit for '{key[:30]}...'")
            return cache_data["value"]
                
        except Exception as e:
            print(f"Error reading cache: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key: The key to set
            value: The value to cache
        """
        cache_file = self._get_cache_file(key)
        
        try:
            # Create cache entry
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "key": key,
                "value": value
            }
            
            # Write to cache file
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"Cached result for '{key[:30]}...'")
                
        except Exception as e:
            print(f"Error writing to cache: {str(e)}")
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        try:
            # Delete all cache files
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, filename))
                    
            print("Cache cleared")
                
        except Exception as e:
            print(f"Error clearing cache: {str(e)}") 