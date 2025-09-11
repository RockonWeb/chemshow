"""
Рефакторинговая версия database.py с устранением дублирования кода
"""
import sqlite3
import logging
import time
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from contextlib import contextmanager

from .advanced_cache import query_cache, cached
from .database_pool import ConnectionPool, DATABASE_INDEXES, create_indexes

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class SearchConfig:
    """Конфигурация для поиска"""
    text_fields: List[str] = field(default_factory=list)
    mass_field: str = "molecular_weight"
    mass_conversion: float = 1.0  # Коэффициент преобразования массы
    additional_filters: Dict[str, Any] = field(default_factory=dict)


class BaseRepository(ABC, Generic[T]):
    """Универсальный базовый репозиторий с общей логикой"""
    
    def __init__(self, db_type: str, pool: ConnectionPool, search_config: SearchConfig):
        self.db_type = db_type
        self.table_name = db_type
        self.pool = pool
        self.search_config = search_config
    
    @contextmanager
    def get_connection(self):
        """Получает соединение из пула"""
        with self.pool.get_connection() as conn:
            yield conn
    
    @cached(ttl=3600)
    def count_all(self) -> int:
        """Подсчет общего количества записей с кэшированием"""
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        
        # Проверяем кэш
        cached_result = query_cache.get(self.db_type, query)
        if cached_result is not None:
            return cached_result
        
        with self.get_connection() as conn:
            cursor = conn.execute(query)
            count = cursor.fetchone()[0]
        
        # Сохраняем в кэш
        query_cache.set(self.db_type, query, None, count)
        return count
    
    def _build_search_conditions(
        self, 
        search_text: str = None,
        mass: float = None,
        tolerance_ppm: int = 1000,
        additional_filters: Dict[str, Any] = None
    ) -> tuple:
        """Строит условия поиска"""
        conditions = []
        params = []
        
        # Текстовый поиск
        if search_text and search_text.strip():
            text_conditions = []
            for field in self.search_config.text_fields:
                text_conditions.append(f"UPPER({field}) LIKE UPPER(?)")
                params.append(f"%{search_text.strip()}%")
            
            if text_conditions:
                conditions.append("(" + " OR ".join(text_conditions) + ")")
        
        # Поиск по массе
        if mass is not None and mass > 0:
            # Применяем коэффициент конвертации
            converted_mass = mass * self.search_config.mass_conversion
            tolerance = converted_mass * tolerance_ppm / 1000000
            mass_min = converted_mass - tolerance
            mass_max = converted_mass + tolerance
            
            conditions.append(f"{self.search_config.mass_field} BETWEEN ? AND ?")
            params.extend([mass_min, mass_max])
        
        # Дополнительные фильтры
        all_filters = {**self.search_config.additional_filters, **(additional_filters or {})}
        for field, value in all_filters.items():
            if value is not None and value != "Все":
                if field.endswith("_like"):
                    actual_field = field[:-5]
                    conditions.append(f"UPPER({actual_field}) LIKE UPPER(?)")
                    params.append(f"%{value}%")
                else:
                    conditions.append(f"{field} = ?")
                    params.append(value)
        
        return conditions, params
    
    def search(
        self,
        query: str = None,
        mass: float = None,
        tolerance_ppm: int = 1000,
        limit: int = None,
        offset: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Универсальный метод поиска"""
        
        # Строим условия поиска
        conditions, params = self._build_search_conditions(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            additional_filters=kwargs
        )
        
        if not conditions:
            return []
        
        where_clause = " AND ".join(conditions)
        sql_query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
            ORDER BY id
            LIMIT ? OFFSET ?
        """
        params.extend([limit or 50, offset])
        
        # Проверяем кэш
        cache_key = f"{self.db_type}:{sql_query}:{params}"
        cached_result = query_cache.get(self.db_type, sql_query, tuple(params))
        if cached_result is not None:
            return cached_result
        
        # Выполняем запрос
        with self.get_connection() as conn:
            cursor = conn.execute(sql_query, params)
            results = [dict(row) for row in cursor.fetchall()]
        
        # Сохраняем в кэш
        query_cache.set(self.db_type, sql_query, tuple(params), results)
        
        return results
    
    def count_search(
        self,
        query: str = None,
        mass: float = None,
        tolerance_ppm: int = 1000,
        **kwargs
    ) -> int:
        """Подсчет результатов поиска"""
        
        # Строим условия поиска
        conditions, params = self._build_search_conditions(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            additional_filters=kwargs
        )
        
        if not conditions:
            return 0
        
        where_clause = " AND ".join(conditions)
        sql_query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_clause}"
        
        # Проверяем кэш
        cached_result = query_cache.get(self.db_type, sql_query, tuple(params))
        if cached_result is not None:
            return cached_result
        
        # Выполняем запрос
        with self.get_connection() as conn:
            cursor = conn.execute(sql_query, params)
            count = cursor.fetchone()[0]
        
        # Сохраняем в кэш
        query_cache.set(self.db_type, sql_query, tuple(params), count)
        
        return count
    
    def find_by_ids(self, ids: List[int]) -> List[Dict[str, Any]]:
        """Батчевый поиск по ID"""
        if not ids:
            return []
        
        placeholders = ','.join('?' * len(ids))
        query = f"SELECT * FROM {self.table_name} WHERE id IN ({placeholders})"
        
        with self.get_connection() as conn:
            cursor = conn.execute(query, ids)
            return [dict(row) for row in cursor.fetchall()]
    
    def optimize_table(self):
        """Оптимизирует таблицу"""
        with self.get_connection() as conn:
            conn.execute(f"ANALYZE {self.table_name}")
            conn.execute("VACUUM")
            logger.info(f"Table {self.table_name} optimized")


class RepositoryFactory:
    """Фабрика для создания репозиториев"""
    
    _repositories = {}
    _pools = {}
    
    @classmethod
    def create_repository(
        cls,
        db_type: str,
        db_path: str,
        search_config: SearchConfig,
        pool_size: int = 5
    ) -> BaseRepository:
        """Создает репозиторий с пулом соединений"""
        
        # Создаем пул если его еще нет
        if db_type not in cls._pools:
            cls._pools[db_type] = ConnectionPool(db_path, pool_size)
            
            # Создаем индексы при первом создании
            if db_type in DATABASE_INDEXES:
                create_indexes(db_path, db_type, DATABASE_INDEXES[db_type])
        
        # Создаем репозиторий если его еще нет
        if db_type not in cls._repositories:
            cls._repositories[db_type] = BaseRepository(
                db_type=db_type,
                pool=cls._pools[db_type],
                search_config=search_config
            )
        
        return cls._repositories[db_type]
    
    @classmethod
    def get_repository(cls, db_type: str) -> Optional[BaseRepository]:
        """Получает существующий репозиторий"""
        return cls._repositories.get(db_type)
    
    @classmethod
    def close_all(cls):
        """Закрывает все пулы соединений"""
        for pool in cls._pools.values():
            pool.close()
        cls._pools.clear()
        cls._repositories.clear()


# Конфигурации для разных типов БД
REPOSITORY_CONFIGS = {
    "metabolites": SearchConfig(
        text_fields=["name", "name_ru", "formula", "class_name"],
        mass_field="exact_mass",
        mass_conversion=1.0
    ),
    "enzymes": SearchConfig(
        text_fields=["name", "name_ru", "ec_number", "family"],
        mass_field="molecular_weight",
        mass_conversion=0.001  # Конвертация Da в kDa
    ),
    "proteins": SearchConfig(
        text_fields=["name", "name_ru", "function", "family"],
        mass_field="molecular_weight",
        mass_conversion=0.001  # Конвертация Da в kDa
    ),
    "carbohydrates": SearchConfig(
        text_fields=["name", "name_ru", "formula", "type"],
        mass_field="exact_mass",
        mass_conversion=1.0
    ),
    "lipids": SearchConfig(
        text_fields=["name", "name_ru", "formula", "type"],
        mass_field="exact_mass",
        mass_conversion=1.0
    )
}


class DatabaseManager:
    """Менеджер для управления всеми репозиториями"""
    
    def __init__(self, db_paths: Dict[str, str]):
        self.db_paths = db_paths
        self.repositories = {}
        
        # Инициализируем репозитории
        for db_type, path in db_paths.items():
            if db_type in REPOSITORY_CONFIGS:
                try:
                    repo = RepositoryFactory.create_repository(
                        db_type=db_type,
                        db_path=path,
                        search_config=REPOSITORY_CONFIGS[db_type]
                    )
                    self.repositories[db_type] = repo
                    logger.info(f"Repository {db_type} initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize repository {db_type}: {e}")
    
    def get_repository(self, db_type: str) -> Optional[BaseRepository]:
        """Получает репозиторий по типу"""
        return self.repositories.get(db_type)
    
    def get_database_stats(self) -> Dict[str, int]:
        """Получает статистику по всем БД"""
        stats = {}
        
        for db_type, repo in self.repositories.items():
            try:
                stats[db_type] = repo.count_all()
            except Exception as e:
                logger.error(f"Error getting stats for {db_type}: {e}")
                stats[db_type] = 0
        
        stats["db_status"] = "healthy" if sum(stats.values()) > 0 else "offline"
        return stats
    
    def search_all(
        self,
        query: str = None,
        mass: float = None,
        tolerance_ppm: int = 1000,
        organism_type: str = None,
        page: int = 1,
        page_size: int = 50
    ) -> Dict[str, Any]:
        """Поиск по всем БД одновременно"""
        results = {}
        offset = (page - 1) * page_size
        
        for db_type, repo in self.repositories.items():
            try:
                # Добавляем фильтр по организму для ферментов и белков
                kwargs = {}
                if organism_type and db_type in ["enzymes", "proteins"]:
                    kwargs["organism_type"] = organism_type
                
                # Выполняем поиск
                data = repo.search(
                    query=query,
                    mass=mass,
                    tolerance_ppm=tolerance_ppm,
                    limit=page_size,
                    offset=offset,
                    **kwargs
                )
                
                # Подсчитываем общее количество
                total = repo.count_search(
                    query=query,
                    mass=mass,
                    tolerance_ppm=tolerance_ppm,
                    **kwargs
                )
                
                results[db_type] = {
                    "data": data,
                    "total": total
                }
                
            except Exception as e:
                logger.error(f"Search error in {db_type}: {e}")
                results[db_type] = {"data": [], "total": 0}
        
        return results
    
    def close(self):
        """Закрывает все соединения"""
        RepositoryFactory.close_all()
