"""
Продвинутая система кэширования с поддержкой Redis и in-memory cache
"""
import time
import json
import hashlib
import logging
from typing import Any, Optional, Dict, Union
from functools import wraps, lru_cache
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)


class LRUCache:
    """Потокобезопасный LRU cache в памяти"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из кэша"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Проверяем TTL
                if time.time() - timestamp < self.ttl:
                    # Перемещаем в конец (most recently used)
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
                else:
                    # Удаляем устаревшее значение
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Добавляет значение в кэш"""
        with self.lock:
            # Удаляем ключ если он существует
            if key in self.cache:
                del self.cache[key]
            
            # Добавляем в конец
            self.cache[key] = (value, time.time())
            
            # Проверяем размер кэша
            if len(self.cache) > self.max_size:
                # Удаляем самый старый элемент
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Очищает кэш"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2f}%",
                "ttl_seconds": self.ttl
            }


class RedisCache:
    """Кэш на основе Redis (если доступен)"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, ttl: int = 3600):
        self.ttl = ttl
        self.redis_client = None
        self.connected = False
        
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=False,
                socket_connect_timeout=2
            )
            # Проверяем подключение
            self.redis_client.ping()
            self.connected = True
            logger.info("Redis cache connected successfully")
        except ImportError:
            logger.warning("Redis not installed, falling back to in-memory cache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, falling back to in-memory cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из Redis"""
        if not self.connected:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Сохраняет значение в Redis"""
        if not self.connected:
            return
        
        try:
            serialized = pickle.dumps(value)
            self.redis_client.setex(
                key,
                ttl or self.ttl,
                serialized
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> bool:
        """Удаляет значение из Redis"""
        if not self.connected:
            return False
        
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Удаляет все ключи по паттерну"""
        if not self.connected:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
        
        return 0


class HybridCache:
    """Гибридный кэш: Redis + in-memory + disk fallback"""
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        memory_size: int = 1000,
        disk_path: Optional[Path] = None,
        ttl: int = 3600
    ):
        # Уровни кэширования
        self.memory_cache = LRUCache(max_size=memory_size, ttl=ttl)
        self.redis_cache = RedisCache(host=redis_host, port=redis_port, ttl=ttl)
        
        # Дисковый кэш как последний уровень
        self.disk_path = disk_path or Path("ui/cache")
        self.disk_path.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Генерирует ключ кэша"""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Получает значение из многоуровневого кэша"""
        
        # Уровень 1: Memory cache
        value = self.memory_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit (memory): {key}")
            return value
        
        # Уровень 2: Redis cache
        value = self.redis_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit (redis): {key}")
            # Сохраняем в memory cache
            self.memory_cache.set(key, value)
            return value
        
        # Уровень 3: Disk cache
        cache_file = self.disk_path / f"{key}.cache"
        if cache_file.exists():
            try:
                # Проверяем TTL
                if time.time() - cache_file.stat().st_mtime < self.ttl:
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                    logger.debug(f"Cache hit (disk): {key}")
                    
                    # Сохраняем в более быстрые уровни
                    self.memory_cache.set(key, value)
                    self.redis_cache.set(key, value)
                    
                    return value
                else:
                    # Удаляем устаревший файл
                    cache_file.unlink()
            except Exception as e:
                logger.error(f"Disk cache read error: {e}")
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Сохраняет значение во все уровни кэша"""
        
        # Сохраняем во все уровни
        self.memory_cache.set(key, value)
        self.redis_cache.set(key, value, ttl)
        
        # Сохраняем на диск
        try:
            cache_file = self.disk_path / f"{key}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Disk cache write error: {e}")
    
    def invalidate(self, key: str) -> None:
        """Инвалидирует кэш на всех уровнях"""
        
        # Удаляем из memory cache
        with self.memory_cache.lock:
            if key in self.memory_cache.cache:
                del self.memory_cache.cache[key]
        
        # Удаляем из Redis
        self.redis_cache.delete(key)
        
        # Удаляем с диска
        cache_file = self.disk_path / f"{key}.cache"
        if cache_file.exists():
            cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        disk_files = list(self.disk_path.glob("*.cache"))
        disk_size = sum(f.stat().st_size for f in disk_files) / (1024 * 1024)  # MB
        
        return {
            "memory": self.memory_cache.get_stats(),
            "redis_connected": self.redis_cache.connected,
            "disk": {
                "files": len(disk_files),
                "size_mb": f"{disk_size:.2f}"
            }
        }


def cached(ttl: int = 3600, cache_type: str = "hybrid"):
    """Декоратор для кэширования результатов функций"""
    
    # Создаем глобальный кэш
    if cache_type == "memory":
        cache = LRUCache(ttl=ttl)
    elif cache_type == "redis":
        cache = RedisCache(ttl=ttl)
    else:
        cache = HybridCache(ttl=ttl)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Генерируем ключ кэша
            cache_key = hashlib.md5(
                f"{func.__name__}:{args}:{kwargs}".encode()
            ).hexdigest()
            
            # Проверяем кэш
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Вызываем функцию
            result = func(*args, **kwargs)
            
            # Сохраняем результат
            cache.set(cache_key, result)
            
            return result
        
        # Добавляем методы управления кэшем
        wrapper.cache = cache
        wrapper.invalidate = lambda: cache.clear() if hasattr(cache, 'clear') else None
        
        return wrapper
    
    return decorator


class QueryCache:
    """Специализированный кэш для запросов к БД"""
    
    def __init__(self, cache: Union[LRUCache, RedisCache, HybridCache]):
        self.cache = cache
    
    def get_query_key(self, db_type: str, query: str, params: tuple = None) -> str:
        """Генерирует ключ для запроса"""
        key_data = {
            "db": db_type,
            "query": query,
            "params": params
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return f"query:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(self, db_type: str, query: str, params: tuple = None) -> Optional[Any]:
        """Получает результат запроса из кэша"""
        key = self.get_query_key(db_type, query, params)
        return self.cache.get(key)
    
    def set(self, db_type: str, query: str, params: tuple, result: Any, ttl: int = 1800) -> None:
        """Сохраняет результат запроса в кэш"""
        key = self.get_query_key(db_type, query, params)
        
        if hasattr(self.cache, 'set'):
            if hasattr(self.cache.set, '__code__') and 'ttl' in self.cache.set.__code__.co_varnames:
                self.cache.set(key, result, ttl=ttl)
            else:
                self.cache.set(key, result)
    
    def invalidate_db(self, db_type: str) -> None:
        """Инвалидирует весь кэш для конкретной БД"""
        pattern = f"query:*{db_type}*"
        
        if hasattr(self.cache, 'clear_pattern'):
            self.cache.clear_pattern(pattern)
        elif hasattr(self.cache, 'redis_cache'):
            self.cache.redis_cache.clear_pattern(pattern)


# Глобальные экземпляры кэшей
memory_cache = LRUCache(max_size=2000, ttl=3600)
hybrid_cache = HybridCache(memory_size=2000, ttl=3600)
query_cache = QueryCache(hybrid_cache)
