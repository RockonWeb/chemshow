"""
Улучшенная система работы с базами данных с пулом соединений
"""
import sqlite3
import logging
import time
import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List, Any, Optional, AsyncGenerator, Generator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import queue

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Пул соединений для SQLite"""
    
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._connections = queue.Queue(maxsize=pool_size)
        self._lock = Lock()
        self._closed = False
        
        # Инициализируем пул
        for _ in range(pool_size):
            conn = self._create_connection()
            self._connections.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Создает новое соединение с оптимизациями"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Оптимизации для SQLite
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=30000000000")  # Memory-mapped I/O
        
        return conn
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Получает соединение из пула"""
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn = None
        try:
            conn = self._connections.get(timeout=10)
            yield conn
        finally:
            if conn:
                # Возвращаем соединение в пул
                self._connections.put(conn)
    
    def close(self):
        """Закрывает все соединения в пуле"""
        with self._lock:
            self._closed = True
            while not self._connections.empty():
                try:
                    conn = self._connections.get_nowait()
                    conn.close()
                except queue.Empty:
                    break


class AsyncDatabaseManager:
    """Асинхронный менеджер базы данных с пулом соединений"""
    
    def __init__(self, db_paths: Dict[str, str], pool_size: int = 5):
        self.db_paths = db_paths
        self.pools = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Создаем пулы для каждой БД
        for db_type, path in db_paths.items():
            if Path(path).exists():
                self.pools[db_type] = ConnectionPool(path, pool_size)
    
    async def execute_query(
        self, 
        db_type: str, 
        query: str, 
        params: tuple = None
    ) -> List[Dict[str, Any]]:
        """Асинхронное выполнение запроса"""
        if db_type not in self.pools:
            raise ValueError(f"Unknown database type: {db_type}")
        
        loop = asyncio.get_event_loop()
        
        def _execute():
            with self.pools[db_type].get_connection() as conn:
                cursor = conn.execute(query, params or ())
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        
        # Выполняем в потоке
        result = await loop.run_in_executor(self.executor, _execute)
        return result
    
    async def execute_many(
        self,
        db_type: str,
        query: str,
        params_list: List[tuple]
    ) -> None:
        """Батчевое выполнение запросов"""
        if db_type not in self.pools:
            raise ValueError(f"Unknown database type: {db_type}")
        
        loop = asyncio.get_event_loop()
        
        def _execute_many():
            with self.pools[db_type].get_connection() as conn:
                conn.executemany(query, params_list)
                conn.commit()
        
        await loop.run_in_executor(self.executor, _execute_many)
    
    async def execute_count(
        self,
        db_type: str,
        query: str,
        params: tuple = None
    ) -> int:
        """Асинхронное выполнение COUNT запроса"""
        if db_type not in self.pools:
            raise ValueError(f"Unknown database type: {db_type}")
        
        loop = asyncio.get_event_loop()
        
        def _execute_count():
            with self.pools[db_type].get_connection() as conn:
                cursor = conn.execute(query, params or ())
                result = cursor.fetchone()
                return result[0] if result else 0
        
        count = await loop.run_in_executor(self.executor, _execute_count)
        return count
    
    def close(self):
        """Закрывает все пулы и executor"""
        for pool in self.pools.values():
            pool.close()
        self.executor.shutdown(wait=True)


class OptimizedRepository:
    """Оптимизированный базовый репозиторий"""
    
    def __init__(self, db_manager: AsyncDatabaseManager, db_type: str):
        self.db_manager = db_manager
        self.db_type = db_type
        self.table_name = db_type
    
    async def find_batch(
        self,
        ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Батчевый поиск по ID"""
        if not ids:
            return []
        
        placeholders = ','.join('?' * len(ids))
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE id IN ({placeholders})
        """
        
        return await self.db_manager.execute_query(
            self.db_type, query, tuple(ids)
        )
    
    async def search_optimized(
        self,
        search_text: str = None,
        mass: float = None,
        tolerance_ppm: int = 1000,
        text_fields: List[str] = None,
        mass_field: str = "molecular_weight",
        limit: int = None,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Оптимизированный поиск с подсчетом"""
        
        conditions = []
        params = []
        
        # Построение условий поиска
        if search_text and search_text.strip():
            text_conditions = []
            for field in text_fields or ["name"]:
                text_conditions.append(f"{field} LIKE ?")
                params.append(f"%{search_text}%")
            
            if text_conditions:
                conditions.append("(" + " OR ".join(text_conditions) + ")")
        
        if mass is not None and mass > 0:
            tolerance = mass * tolerance_ppm / 1000000
            mass_min = mass - tolerance
            mass_max = mass + tolerance
            conditions.append(f"{mass_field} BETWEEN ? AND ?")
            params.extend([mass_min, mass_max])
        
        if not conditions:
            return {"data": [], "total": 0}
        
        where_clause = " AND ".join(conditions)
        
        # Параллельное выполнение запроса данных и подсчета
        data_query = f"""
            SELECT * FROM {self.table_name}
            WHERE {where_clause}
            LIMIT ? OFFSET ?
        """
        
        count_query = f"""
            SELECT COUNT(*) FROM {self.table_name}
            WHERE {where_clause}
        """
        
        # Выполняем оба запроса параллельно
        data_task = self.db_manager.execute_query(
            self.db_type, 
            data_query, 
            tuple(params + [limit or 50, offset])
        )
        
        count_task = self.db_manager.execute_count(
            self.db_type,
            count_query,
            tuple(params)
        )
        
        data, total = await asyncio.gather(data_task, count_task)
        
        return {
            "data": data,
            "total": total
        }


# Создаем индексы для оптимизации запросов
def create_indexes(db_path: str, table_name: str, indexes: Dict[str, List[str]]):
    """Создает индексы для оптимизации запросов"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for index_name, columns in indexes.items():
        columns_str = ", ".join(columns)
        try:
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_{index_name}
                ON {table_name} ({columns_str})
            """)
            logger.info(f"Created index idx_{table_name}_{index_name}")
        except sqlite3.Error as e:
            logger.error(f"Error creating index {index_name}: {e}")
    
    conn.commit()
    conn.close()


def optimize_database(db_path: str):
    """Оптимизирует базу данных"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Анализ таблиц для оптимизации планов запросов
        cursor.execute("ANALYZE")
        
        # Очистка неиспользуемого пространства
        cursor.execute("VACUUM")
        
        # Проверка целостности
        cursor.execute("PRAGMA integrity_check")
        
        logger.info(f"Database optimized: {db_path}")
    except sqlite3.Error as e:
        logger.error(f"Error optimizing database: {e}")
    finally:
        conn.close()


# Индексы для каждого типа БД
DATABASE_INDEXES = {
    "metabolites": {
        "name_search": ["name", "name_ru"],
        "mass_search": ["exact_mass"],
        "formula_search": ["formula"],
        "class_search": ["class_name"],
        "combined": ["name", "exact_mass"]
    },
    "enzymes": {
        "name_search": ["name", "name_ru"],
        "ec_search": ["ec_number"],
        "organism_search": ["organism", "organism_type"],
        "family_search": ["family"],
        "mass_search": ["molecular_weight"]
    },
    "proteins": {
        "name_search": ["name", "name_ru"],
        "function_search": ["function"],
        "organism_search": ["organism", "organism_type"],
        "family_search": ["family"],
        "mass_search": ["molecular_weight"]
    },
    "carbohydrates": {
        "name_search": ["name", "name_ru"],
        "type_search": ["type"],
        "mass_search": ["exact_mass"],
        "formula_search": ["formula"]
    },
    "lipids": {
        "name_search": ["name", "name_ru"],
        "type_search": ["type"],
        "mass_search": ["exact_mass"],
        "formula_search": ["formula"]
    }
}
