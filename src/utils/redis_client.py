#!/usr/bin/env python3
"""
Redis Client - Connection and utilities
"""

import json
from fnmatch import fnmatch
from typing import Any, Optional, Dict

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    redis = None

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__, "redis.log")

class _InMemoryRedis:
    """Minimal Redis-like stub for offline usage."""

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


class RedisClient:
    """Redis client wrapper with connection pooling"""
    
    _instance = None
    _redis = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._store: Dict[str, Any] = {}
        self._use_stub = False
        self._connect()
    
    def _connect(self):
        """Establish Redis connection"""
        try:
            if redis is None:
                raise ImportError("redis package not installed")

            self._redis = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD if Config.REDIS_PASSWORD else None,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                connection_pool=redis.ConnectionPool(
                    host=Config.REDIS_HOST,
                    port=Config.REDIS_PORT,
                    db=Config.REDIS_DB,
                    max_connections=10,
                    retry_on_timeout=True
                )
            )
            
            # Test connection
            self._redis.ping()
            logger.info(f"✓ Redis connected: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        except Exception as e:
            logger.warning(f"✗ Redis connection unavailable ({e}). Using in-memory cache.")
            self._redis = _InMemoryRedis()
            self._use_stub = True
    
    @staticmethod
    def get_instance() -> 'RedisClient':
        """Get singleton instance"""
        if RedisClient._instance is None:
            RedisClient()
        return RedisClient._instance
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            if self._use_stub:
                value = self._store.get(key)
            else:
                value = self._redis.get(key)

            if value is None:
                return None

            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            return value
        except json.JSONDecodeError:
            return value
        except Exception as e:
            logger.error(f"Redis GET error ({key}): {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 86400) -> bool:
        """Set value in Redis with TTL"""
        try:
            json_value = json.dumps(value) if isinstance(value, (dict, list)) else value

            if self._use_stub:
                self._store[key] = json.loads(json_value) if isinstance(json_value, str) else json_value
            else:
                self._redis.setex(key, ttl, json_value)
            return True
        except Exception as e:
            logger.error(f"Redis SET error ({key}): {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        try:
            if self._use_stub:
                self._store.pop(key, None)
            else:
                self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis DELETE error ({key}): {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if self._use_stub:
                return key in self._store
            return self._redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error ({key}): {e}")
            return False
    
    def keys(self, pattern: str) -> list:
        """Get keys matching pattern"""
        try:
            if self._use_stub:
                return [key for key in self._store if fnmatch(key, pattern)]
            return self._redis.keys(pattern)
        except Exception as e:
            logger.error(f"Redis KEYS error ({pattern}): {e}")
            return []
    
    def incr(self, key: str) -> int:
        """Increment counter"""
        try:
            if self._use_stub:
                current = int(self._store.get(key, 0))
                current += 1
                self._store[key] = current
                return current
            return self._redis.incr(key)
        except Exception as e:
            logger.error(f"Redis INCR error ({key}): {e}")
            return 0
    
    def ttl(self, key: str) -> int:
        """Get TTL for key"""
        try:
            if self._use_stub:
                return -1
            return self._redis.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL error ({key}): {e}")
            return -1
    
    def flush_all(self):
        """Flush all Redis data (DANGER!)"""
        try:
            if self._use_stub:
                self._store.clear()
            else:
                self._redis.flushall()
            logger.warning("Redis FLUSHALL executed")
        except Exception as e:
            logger.error(f"Redis FLUSHALL error: {e}")

    def get_info(self) -> Dict:
        """Get Redis info"""
        try:
            if self._use_stub:
                return {
                    'mode': 'in-memory',
                    'keys': len(self._store)
                }
            return self._redis.info()
        except Exception as e:
            logger.error(f"Redis INFO error: {e}")
            return {}

    def close(self):
        """Close Redis connection"""
        if self._redis:
            try:
                self._redis.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Redis close error: {e}")


# Convenience instance
redis_client = RedisClient.get_instance()