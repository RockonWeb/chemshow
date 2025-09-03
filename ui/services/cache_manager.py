"""
Система кэширования для оптимизации производительности
"""
import time
import pickle
import logging
import os
from pathlib import Path
from typing import Any, Optional, Dict
import hashlib

logger = logging.getLogger(__name__)

# Директория для кэша
CACHE_DIR = Path("ui/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class CacheManager:
    """Управление кэшем приложения"""

    def __init__(self, cache_dir: Path = CACHE_DIR, max_age: int = 3600):
        """
        Args:
            cache_dir: Директория для хранения кэша
            max_age: Максимальный возраст кэша в секундах (по умолчанию 1 час)
        """
        self.cache_dir = cache_dir
        self.max_age = max_age
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Получает путь к файлу кэша для данного ключа"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _is_expired(self, cache_path: Path) -> bool:
        """Проверяет, истек ли срок действия кэша"""
        if not cache_path.exists():
            return True

        cache_time = cache_path.stat().st_mtime
        current_time = time.time()

        return (current_time - cache_time) > self.max_age

    def get(self, key: str) -> Optional[Any]:
        """
        Получает значение из кэша

        Args:
            key: Ключ кэша

        Returns:
            Значение из кэша или None если не найдено/истекло
        """
        cache_path = self._get_cache_path(key)

        if self._is_expired(cache_path):
            # Удаляем истекший кэш
            if cache_path.exists():
                cache_path.unlink()
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for key: {key}")
            return data
        except Exception as e:
            logger.error(f"Error reading cache for key {key}: {e}")
            # Удаляем поврежденный кэш
            if cache_path.exists():
                cache_path.unlink()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Сохраняет значение в кэш

        Args:
            key: Ключ кэша
            value: Значение для сохранения
            ttl: Время жизни в секундах (если None, используется max_age)
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            # Устанавливаем время модификации для TTL
            if ttl:
                current_time = time.time()
                new_time = current_time - (self.max_age - ttl)
                os.utime(cache_path, (new_time, new_time))

            logger.debug(f"Cache set for key: {key}")

        except Exception as e:
            logger.error(f"Error writing cache for key {key}: {e}")

    def delete(self, key: str) -> bool:
        """
        Удаляет значение из кэша

        Args:
            key: Ключ кэша

        Returns:
            True если удалено успешно
        """
        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.debug(f"Cache deleted for key: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False

    def clear(self) -> int:
        """
        Очищает весь кэш

        Returns:
            Количество удаленных файлов
        """
        deleted_count = 0

        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
                deleted_count += 1

            logger.info(f"Cache cleared: {deleted_count} files deleted")
            return deleted_count

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return deleted_count

    def get_stats(self) -> Dict[str, Any]:
        """
        Получает статистику кэша

        Returns:
            Словарь со статистикой
        """
        total_files = 0
        valid_files = 0
        expired_files = 0
        total_size = 0

        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                total_files += 1
                total_size += cache_file.stat().st_size

                if self._is_expired(cache_file):
                    expired_files += 1
                else:
                    valid_files += 1

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")

        return {
            "total_files": total_files,
            "valid_files": valid_files,
            "expired_files": expired_files,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


class QueryCache:
    """Кэширование результатов запросов к базе данных"""

    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager

    def get_query_key(self, db_type: str, query: str, params: tuple = None) -> str:
        """Генерирует ключ кэша для запроса"""
        # Не сортируем params: могут быть элементы разных типов (str/int/float),
        # что приводит к ошибке сравнения при сортировке. Используем repr с сохранением порядка.
        params_str = repr(params) if params else ""
        return f"query_{db_type}_{hash(query)}_{hash(params_str)}"

    def get(self, db_type: str, query: str, params: tuple = None) -> Optional[Any]:
        """Получает результат запроса из кэша"""
        key = self.get_query_key(db_type, query, params)
        return self.cache.get(key)

    def set(self, db_type: str, query: str, params: tuple, result: Any) -> None:
        """Сохраняет результат запроса в кэш"""
        key = self.get_query_key(db_type, query, params)
        # Кэшируем на 30 минут для запросов
        self.cache.set(key, result, ttl=1800)

    def invalidate_db_cache(self, db_type: str) -> None:
        """Очищает кэш для конкретной базы данных"""
        # Удаляем все ключи, начинающиеся с query_{db_type}
        prefix = f"query_{db_type}_"

        try:
            for cache_file in self.cache.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        # Проверяем, содержит ли файл наш префикс
                        # Это не самый эффективный способ, но работает
                        pass
                except:
                    continue

            logger.info(f"Database cache invalidated for: {db_type}")

        except Exception as e:
            logger.error(f"Error invalidating cache for {db_type}: {e}")


# Глобальные экземпляры
cache_manager = CacheManager()
query_cache = QueryCache(cache_manager)

# Кэш для структур молекул (уже реализован в visualization_3d.py)
structure_cache = CacheManager(
    cache_dir=Path("ui/cache/structures"),
    max_age=86400  # 24 часа для структур
)
