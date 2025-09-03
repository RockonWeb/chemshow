"""
Абстракция для работы с базами данных с защитой от SQL-инъекций
"""
import sqlite3
import logging
import time
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Generator
import sys
from pathlib import Path

# Пути к базам данных
DATABASE_PATHS = {
    "metabolites": "data/metabolites.db",
    "enzymes": "data/enzymes.db",
    "proteins": "data/proteins.db",
    "carbohydrates": "data/carbohydrates.db",
    "lipids": "data/lipids.db",
}

# Импортируем кэш-менеджер
try:
    from cache_manager import query_cache
except ImportError:
    # Если импорт не удался, создаем заглушку
    class DummyCache:
        def get(self, *args): return None
        def set(self, *args): pass
    query_cache = DummyCache()

# Мониторинг производительности - импортируем только при использовании
_performance_monitor = None
def get_performance_monitor():
    """Ленивый импорт монитора производительности"""
    global _performance_monitor
    if _performance_monitor is None:
        try:
            from performance_monitor import performance_monitor
            _performance_monitor = performance_monitor
        except ImportError:
            class DummyMonitor:
                def log_query(self, *args, **kwargs): pass
            _performance_monitor = DummyMonitor()
    return _performance_monitor

logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Ошибка подключения к базе данных"""
    pass


class DatabaseQueryError(Exception):
    """Ошибка выполнения запроса"""
    pass


class DatabaseManager:
    """Управление подключениями к базам данных"""

    def __init__(self, db_paths: Dict[str, str]):
        self.db_paths = db_paths
        self._connections = {}

    @contextmanager
    def get_connection(self, db_type: str) -> Generator[sqlite3.Connection, None, None]:
        """Контекстный менеджер для безопасного подключения к БД"""
        if db_type not in self.db_paths:
            raise DatabaseConnectionError(f"Неизвестный тип базы данных: {db_type}")

        db_path = self.db_paths[db_type]

        try:
            # Создаем новое подключение для каждого запроса
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")  # Улучшает производительность
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

            # Проверяем существование основной таблицы
            self._validate_database(conn, db_type)

            yield conn

        except sqlite3.Error as e:
            logger.error(f"Ошибка работы с БД {db_type}: {e}")
            raise DatabaseQueryError(f"Ошибка базы данных: {e}")
        finally:
            if 'conn' in locals():
                conn.close()

    def _validate_database(self, conn: sqlite3.Connection, db_type: str) -> None:
        """Проверка корректности базы данных"""
        table_names = {
            "metabolites": "metabolites",
            "enzymes": "enzymes",
            "proteins": "proteins",
            "carbohydrates": "carbohydrates",
            "lipids": "lipids"
        }

        table_name = table_names.get(db_type)
        if not table_name:
            raise DatabaseConnectionError(f"Неизвестный тип базы данных: {db_type}")

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )

        if not cursor.fetchone():
            raise DatabaseConnectionError(f"Таблица '{table_name}' не найдена в базе данных {db_type}")

    def execute_query(self, db_type: str, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Безопасное выполнение SELECT запроса с кэшированием"""
        start_time = time.time()
        cache_hit = False

        # Проверяем кэш для SELECT запросов
        if query.strip().upper().startswith('SELECT'):
            cached_result = query_cache.get(db_type, query, params)
            if cached_result is not None:
                logger.debug(f"Query cache hit for {db_type}")
                cache_hit = True
                duration = time.time() - start_time
                get_performance_monitor().log_query(
                    db_type=db_type,
                    query_type="SELECT",
                    duration=duration,
                    records_count=len(cached_result),
                    cache_hit=True
                )
                return cached_result

        # Выполняем запрос
        with self.get_connection(db_type) as conn:
            cursor = conn.execute(query, params or ())
            rows = cursor.fetchall()
            result = [dict(row) for row in rows]

            # Кэшируем результат для SELECT запросов
            if query.strip().upper().startswith('SELECT'):
                query_cache.set(db_type, query, params, result)
                logger.debug(f"Query result cached for {db_type}")

            # Логируем производительность
            duration = time.time() - start_time
            get_performance_monitor().log_query(
                db_type=db_type,
                query_type=query.split()[0].upper(),
                duration=duration,
                records_count=len(result),
                cache_hit=cache_hit
            )

            return result

    def execute_count_query(self, db_type: str, query: str, params: tuple = None) -> int:
        """Безопасное выполнение COUNT запроса с кэшированием"""
        start_time = time.time()
        cache_hit = False

        # Проверяем кэш
        cached_result = query_cache.get(db_type, query, params)
        if cached_result is not None:
            logger.debug(f"Count query cache hit for {db_type}")
            cache_hit = True
            duration = time.time() - start_time
            get_performance_monitor().log_query(
                db_type=db_type,
                query_type="COUNT",
                duration=duration,
                records_count=cached_result,
                cache_hit=True
            )
            return cached_result

        # Выполняем запрос
        with self.get_connection(db_type) as conn:
            cursor = conn.execute(query, params or ())
            result = cursor.fetchone()
            count = result[0] if result else 0

            # Кэшируем результат
            query_cache.set(db_type, query, params, count)
            logger.debug(f"Count query result cached for {db_type}")

            # Логируем производительность
            duration = time.time() - start_time
            get_performance_monitor().log_query(
                db_type=db_type,
                query_type="COUNT",
                duration=duration,
                records_count=count,
                cache_hit=cache_hit
            )

            return count


# Глобальный экземпляр менеджера БД
db_manager = DatabaseManager(DATABASE_PATHS)


class BaseRepository:
    """Базовый класс для репозиториев"""

    def __init__(self, db_type: str):
        self.db_type = db_type
        self.table_name = db_type  # Предполагаем, что имя таблицы совпадает с типом БД

    def count_all(self) -> int:
        """Подсчет общего количества записей"""
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        return db_manager.execute_count_query(self.db_type, query)

    def find_by_text(self, search_text: str, fields: List[str], limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Поиск по текстовым полям"""
        if not search_text or not search_text.strip():
            return []

        # Конвертация limit и offset для безопасности
        try:
            if isinstance(limit, str):
                limit = int(limit) if limit.strip() else None
            elif limit is not None:
                limit = int(limit)
        except (ValueError, TypeError):
            limit = None

        try:
            if isinstance(offset, str):
                offset = int(offset) if offset.strip() else 0
            elif offset is not None:
                offset = int(offset)
        except (ValueError, TypeError):
            offset = 0

        # Форматируем поисковый запрос
        formatted_query = search_text.strip().capitalize()

        # Создаем условия поиска
        conditions = []
        params = []

        for field in fields:
            conditions.append(f"{field} LIKE ?")
            params.append(f"%{formatted_query}%")

        where_clause = " OR ".join(conditions)

        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        return db_manager.execute_query(self.db_type, query, tuple(params))

    def find_by_mass(self, mass: float = None, tolerance_ppm: int = 1000, mass_field: str = "molecular_weight",
                    limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Поиск по массе с допуском"""
        # Конвертация типов для безопасности
        try:
            if isinstance(mass, str):
                mass = float(mass) if mass.strip() else None
            elif mass is not None:
                mass = float(mass)
        except (ValueError, TypeError):
            mass = None

        try:
            if isinstance(tolerance_ppm, str):
                tolerance_ppm = int(tolerance_ppm) if tolerance_ppm.strip() else 1000
            elif tolerance_ppm is not None:
                tolerance_ppm = int(tolerance_ppm)
        except (ValueError, TypeError):
            tolerance_ppm = 1000

        # Конвертация limit и offset для безопасности
        try:
            if isinstance(limit, str):
                limit = int(limit) if limit.strip() else None
            elif limit is not None:
                limit = int(limit)
        except (ValueError, TypeError):
            limit = None

        try:
            if isinstance(offset, str):
                offset = int(offset) if offset.strip() else 0
            elif offset is not None:
                offset = int(offset)
        except (ValueError, TypeError):
            offset = 0

        # Проверяем, что масса корректна и больше 0
        if mass is None or not isinstance(mass, (int, float)) or mass <= 0:
            return []

        # Рассчитываем границы допуска
        tolerance = mass * tolerance_ppm / 1000000
        mass_min = mass - tolerance
        mass_max = mass + tolerance

        query = f"SELECT * FROM {self.table_name} WHERE {mass_field} BETWEEN ? AND ?"
        params = [mass_min, mass_max]

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        return db_manager.execute_query(self.db_type, query, tuple(params))

    def find_combined(self, search_text: str = None, mass: float = None, tolerance_ppm: int = 1000,
                     text_fields: List[str] = None, mass_field: str = "molecular_weight",
                     additional_filters: Dict[str, Any] = None,
                     limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        """Комбинированный поиск по тексту и массе"""
        # Конвертация типов для безопасности
        try:
            if isinstance(mass, str):
                mass = float(mass) if mass.strip() else None
            elif mass is not None:
                mass = float(mass)
        except (ValueError, TypeError):
            mass = None

        try:
            if isinstance(tolerance_ppm, str):
                tolerance_ppm = int(tolerance_ppm) if tolerance_ppm.strip() else 1000
            elif tolerance_ppm is not None:
                tolerance_ppm = int(tolerance_ppm)
        except (ValueError, TypeError):
            tolerance_ppm = 1000

        # Конвертация limit и offset для безопасности
        try:
            if isinstance(limit, str):
                limit = int(limit) if limit.strip() else None
            elif limit is not None:
                limit = int(limit)
        except (ValueError, TypeError):
            limit = None

        try:
            if isinstance(offset, str):
                offset = int(offset) if offset.strip() else 0
            elif offset is not None:
                offset = int(offset)
        except (ValueError, TypeError):
            offset = 0

        if search_text is None:
            search_text = ""

        conditions = []
        params = []

        # Условия поиска по тексту (только если есть текст для поиска)
        if search_text and search_text.strip():
            formatted_query = search_text.strip().capitalize()
            text_conditions = []

            for field in text_fields or ["name"]:
                text_conditions.append(f"{field} LIKE ?")
                params.append(f"%{formatted_query}%")

            if text_conditions:
                conditions.append("(" + " OR ".join(text_conditions) + ")")

        # Условия поиска по массе (только если масса корректно конвертирована)
        if mass is not None and isinstance(mass, (int, float)) and mass > 0:
            tolerance = mass * tolerance_ppm / 1000000
            mass_min = mass - tolerance
            mass_max = mass + tolerance
            conditions.append(f"{mass_field} BETWEEN ? AND ?")
            params.extend([mass_min, mass_max])

        # Дополнительные фильтры
        if additional_filters:
            for field, value in additional_filters.items():
                if value is not None:
                    if field.endswith("_like"):
                        actual_field = field[:-5]  # Убираем "_like"
                        conditions.append(f"{actual_field} LIKE ?")
                        params.append(f"%{value}%")
                    else:
                        conditions.append(f"{field} = ?")
                        params.append(value)

        if not conditions:
            return []

        where_clause = " AND ".join(conditions)
        query = f"SELECT * FROM {self.table_name} WHERE {where_clause}"

        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        return db_manager.execute_query(self.db_type, query, tuple(params))

    def count_combined(self, search_text: str = None, mass: float = None, tolerance_ppm: int = 1000,
                      text_fields: List[str] = None, mass_field: str = "molecular_weight",
                      additional_filters: Dict[str, Any] = None) -> int:
        """Подсчет количества результатов комбинированного поиска"""
        # Конвертация типов для безопасности
        try:
            if isinstance(mass, str):
                mass = float(mass) if mass.strip() else None
            elif mass is not None:
                mass = float(mass)
        except (ValueError, TypeError):
            mass = None

        try:
            if isinstance(tolerance_ppm, str):
                tolerance_ppm = int(tolerance_ppm) if tolerance_ppm.strip() else 1000
            elif tolerance_ppm is not None:
                tolerance_ppm = int(tolerance_ppm)
        except (ValueError, TypeError):
            tolerance_ppm = 1000

        if search_text is None:
            search_text = ""

        conditions = []
        params = []

        # Условия поиска по тексту (только если есть текст для поиска)
        if search_text and search_text.strip():
            formatted_query = search_text.strip().capitalize()
            text_conditions = []

            for field in text_fields or ["name"]:
                text_conditions.append(f"{field} LIKE ?")
                params.append(f"%{formatted_query}%")

            if text_conditions:
                conditions.append("(" + " OR ".join(text_conditions) + ")")

        # Условия поиска по массе (только если масса корректно конвертирована)
        if mass is not None and isinstance(mass, (int, float)) and mass > 0:
            tolerance = mass * tolerance_ppm / 1000000
            mass_min = mass - tolerance
            mass_max = mass + tolerance
            conditions.append(f"{mass_field} BETWEEN ? AND ?")
            params.extend([mass_min, mass_max])

        # Дополнительные фильтры
        if additional_filters:
            for field, value in additional_filters.items():
                if value is not None:
                    if field.endswith("_like"):
                        actual_field = field[:-5]
                        conditions.append(f"{actual_field} LIKE ?")
                        params.append(f"%{value}%")
                    else:
                        conditions.append(f"{field} = ?")
                        params.append(value)

        if not conditions:
            return 0

        where_clause = " AND ".join(conditions)
        query = f"SELECT COUNT(*) FROM {self.table_name} WHERE {where_clause}"

        return db_manager.execute_count_query(self.db_type, query, tuple(params))


# Специализированные репозитории
class MetaboliteRepository(BaseRepository):
    def __init__(self):
        super().__init__("metabolites")

    def search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
               limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        return self.find_combined(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "formula", "class_name"],
            mass_field="exact_mass",
            limit=limit,
            offset=offset
        )

    def count_search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000) -> int:
        return self.count_combined(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "formula", "class_name"],
            mass_field="exact_mass"
        )


class EnzymeRepository(BaseRepository):
    def __init__(self):
        super().__init__("enzymes")

    def search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
               organism_type: str = None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        # Конвертируем массу из Da в kDa для ферментов
        mass_kda = mass / 1000 if mass else None

        filters = {}
        if organism_type and organism_type != "Все":
            filters["organism_type_like"] = organism_type

        return self.find_combined(
            search_text=query,
            mass=mass_kda,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "ec_number", "family"],
            mass_field="molecular_weight",
            additional_filters=filters,
            limit=limit,
            offset=offset
        )

    def count_search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
                    organism_type: str = None) -> int:
        mass_kda = mass / 1000 if mass else None

        filters = {}
        if organism_type and organism_type != "Все":
            filters["organism_type_like"] = organism_type

        return self.count_combined(
            search_text=query,
            mass=mass_kda,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "ec_number", "family"],
            mass_field="molecular_weight",
            additional_filters=filters
        )


class ProteinRepository(BaseRepository):
    def __init__(self):
        super().__init__("proteins")

    def search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
               organism_type: str = None, limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        # Конвертируем массу из Da в kDa для белков
        mass_kda = mass / 1000 if mass else None

        filters = {}
        if organism_type and organism_type != "Все":
            filters["organism_type_like"] = organism_type

        return self.find_combined(
            search_text=query,
            mass=mass_kda,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "function", "family"],
            mass_field="molecular_weight",
            additional_filters=filters,
            limit=limit,
            offset=offset
        )

    def count_search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
                    organism_type: str = None) -> int:
        mass_kda = mass / 1000 if mass else None

        filters = {}
        if organism_type and organism_type != "Все":
            filters["organism_type_like"] = organism_type

        return self.count_combined(
            search_text=query,
            mass=mass_kda,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "function", "family"],
            mass_field="molecular_weight",
            additional_filters=filters
        )


class CarbohydrateRepository(BaseRepository):
    def __init__(self):
        super().__init__("carbohydrates")

    def search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
               limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        return self.find_combined(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "formula", "type"],
            mass_field="exact_mass",
            limit=limit,
            offset=offset
        )

    def count_search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000) -> int:
        return self.count_combined(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "formula", "type"],
            mass_field="exact_mass"
        )


class LipidRepository(BaseRepository):
    def __init__(self):
        super().__init__("lipids")

    def search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000,
               limit: int = None, offset: int = 0) -> List[Dict[str, Any]]:
        return self.find_combined(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "formula", "type"],
            mass_field="exact_mass",
            limit=limit,
            offset=offset
        )

    def count_search(self, query: str = None, mass: float = None, tolerance_ppm: int = 1000) -> int:
        return self.count_combined(
            search_text=query,
            mass=mass,
            tolerance_ppm=tolerance_ppm,
            text_fields=["name", "name_ru", "formula", "type"],
            mass_field="exact_mass"
        )


# Глобальные экземпляры репозиториев
metabolite_repo = MetaboliteRepository()
enzyme_repo = EnzymeRepository()
protein_repo = ProteinRepository()
carbohydrate_repo = CarbohydrateRepository()
lipid_repo = LipidRepository()


def get_database_stats() -> Dict[str, int]:
    """Получение статистики по всем базам данных"""
    stats = {}
    try:
        stats["metabolites"] = metabolite_repo.count_all()
    except Exception as e:
        logger.error(f"Ошибка подсчета метаболитов: {e}")
        stats["metabolites"] = 0

    try:
        stats["enzymes"] = enzyme_repo.count_all()
    except Exception as e:
        logger.error(f"Ошибка подсчета ферментов: {e}")
        stats["enzymes"] = 0

    try:
        stats["proteins"] = protein_repo.count_all()
    except Exception as e:
        logger.error(f"Ошибка подсчета белков: {e}")
        stats["proteins"] = 0

    try:
        stats["carbohydrates"] = carbohydrate_repo.count_all()
    except Exception as e:
        logger.error(f"Ошибка подсчета углеводов: {e}")
        stats["carbohydrates"] = 0

    try:
        stats["lipids"] = lipid_repo.count_all()
    except Exception as e:
        logger.error(f"Ошибка подсчета липидов: {e}")
        stats["lipids"] = 0

    # Определяем статус
    total_records = sum(stats.values())
    stats["db_status"] = "healthy" if total_records > 0 else "offline"

    return stats
