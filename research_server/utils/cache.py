import json
import time
from pathlib import Path
from typing import Any, Optional
from utils.hashing import hash_string
from utils.logging import setup_logger

logger = setup_logger(__name__)

class Cache:
    """Simple file-based cache with expiry."""
    
    def __init__(self, cache_dir: Path, expiry_seconds: int = 86400):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_seconds: Time in seconds before cache entries expire
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expiry_seconds = expiry_seconds
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for a cache key."""
        key_hash = hash_string(key)
        return self.cache_dir / f"{key_hash}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Check expiry
            if time.time() - data['timestamp'] > self.expiry_seconds:
                logger.info(f"Cache expired for key: {key[:50]}...")
                cache_path.unlink()
                return None
            
            logger.info(f"Cache hit for key: {key[:50]}...")
            return data['value']
        
        except (json.JSONDecodeError, KeyError, IOError) as e:
            logger.warning(f"Cache read error for {key[:50]}...: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
        """
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                'timestamp': time.time(),
                'value': value
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Cached value for key: {key[:50]}...")
        
        except (TypeError, IOError) as e:
            logger.error(f"Cache write error for {key[:50]}...: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info("Cache cleared")
