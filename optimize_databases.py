"""
Скрипт для оптимизации всех баз данных
Создает индексы, анализирует таблицы и оптимизирует производительность
"""
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Определение индексов для каждой БД
DATABASE_INDEXES = {
    "metabolites": {
        "name_search": ["name", "name_ru"],
        "mass_search": ["exact_mass"],
        "formula_search": ["formula"],
        "class_search": ["class_name"],
        "combined_search": ["name", "exact_mass"],
        "full_text": ["name", "name_ru", "formula", "class_name"]
    },
    "enzymes": {
        "name_search": ["name", "name_ru"],
        "ec_search": ["ec_number"],
        "organism_search": ["organism"],
        "organism_type_search": ["organism_type"],
        "family_search": ["family"],
        "mass_search": ["molecular_weight"],
        "combined_search": ["name", "molecular_weight"]
    },
    "proteins": {
        "name_search": ["name", "name_ru"],
        "function_search": ["function"],
        "organism_search": ["organism"],
        "organism_type_search": ["organism_type"],
        "family_search": ["family"],
        "mass_search": ["molecular_weight"],
        "combined_search": ["name", "molecular_weight"]
    },
    "carbohydrates": {
        "name_search": ["name", "name_ru"],
        "type_search": ["type"],
        "mass_search": ["exact_mass"],
        "formula_search": ["formula"],
        "combined_search": ["name", "exact_mass"]
    },
    "lipids": {
        "name_search": ["name", "name_ru"],
        "type_search": ["type"],
        "mass_search": ["exact_mass"],
        "formula_search": ["formula"],
        "combined_search": ["name", "exact_mass"]
    }
}


def create_indexes(conn: sqlite3.Connection, table_name: str, indexes: Dict[str, List[str]]):
    """Создает индексы для таблицы"""
    cursor = conn.cursor()
    created_count = 0
    
    for index_name, columns in indexes.items():
        columns_str = ", ".join(columns)
        index_full_name = f"idx_{table_name}_{index_name}"
        
        try:
            # Проверяем, существует ли индекс
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name=?
            """, (index_full_name,))
            
            if cursor.fetchone():
                logger.info(f"  Index {index_full_name} already exists")
            else:
                # Создаем индекс
                cursor.execute(f"""
                    CREATE INDEX {index_full_name}
                    ON {table_name} ({columns_str})
                """)
                logger.info(f"  ✓ Created index {index_full_name}")
                created_count += 1
                
        except sqlite3.Error as e:
            logger.error(f"  ✗ Error creating index {index_full_name}: {e}")
    
    return created_count


def optimize_database(db_path: Path, db_type: str):
    """Оптимизирует отдельную базу данных"""
    if not db_path.exists():
        logger.warning(f"Database {db_path} not found, skipping...")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimizing {db_type} database: {db_path}")
    logger.info(f"{'='*60}")
    
    conn = sqlite3.connect(str(db_path))
    
    try:
        # 1. Создаем индексы
        logger.info("Creating indexes...")
        if db_type in DATABASE_INDEXES:
            created = create_indexes(conn, db_type, DATABASE_INDEXES[db_type])
            logger.info(f"  Created {created} new indexes")
        
        # 2. Устанавливаем оптимальные PRAGMA настройки
        logger.info("Setting optimal PRAGMA settings...")
        pragmas = [
            ("journal_mode", "WAL"),           # Write-Ahead Logging для лучшей конкуренции
            ("synchronous", "NORMAL"),         # Баланс между производительностью и надежностью
            ("cache_size", "-64000"),          # 64MB кэш в памяти
            ("temp_store", "MEMORY"),          # Временные данные в памяти
            ("mmap_size", "30000000000"),      # Memory-mapped I/O
            ("page_size", "4096"),             # Оптимальный размер страницы
            ("auto_vacuum", "INCREMENTAL")     # Автоматическая очистка
        ]
        
        for pragma_name, pragma_value in pragmas:
            try:
                if pragma_name == "page_size":
                    # page_size можно изменить только перед VACUUM
                    current = conn.execute(f"PRAGMA {pragma_name}").fetchone()[0]
                    if current != int(pragma_value):
                        conn.execute(f"PRAGMA {pragma_name} = {pragma_value}")
                        logger.info(f"  ✓ Set {pragma_name} = {pragma_value} (will apply after VACUUM)")
                else:
                    conn.execute(f"PRAGMA {pragma_name} = {pragma_value}")
                    logger.info(f"  ✓ Set {pragma_name} = {pragma_value}")
            except Exception as e:
                logger.warning(f"  ⚠ Could not set {pragma_name}: {e}")
        
        # 3. Анализируем таблицы для оптимизатора запросов
        logger.info("Analyzing tables...")
        conn.execute("ANALYZE")
        logger.info("  ✓ Tables analyzed")
        
        # 4. Получаем статистику
        logger.info("Database statistics:")
        
        # Размер БД
        cursor = conn.cursor()
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        size_mb = size_bytes / (1024 * 1024)
        logger.info(f"  Database size: {size_mb:.2f} MB")
        
        # Количество записей
        cursor.execute(f"SELECT COUNT(*) FROM {db_type}")
        record_count = cursor.fetchone()[0]
        logger.info(f"  Records count: {record_count:,}")
        
        # Количество индексов
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'")
        index_count = cursor.fetchone()[0]
        logger.info(f"  Indexes count: {index_count}")
        
        # 5. Проверка целостности
        logger.info("Checking integrity...")
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        if integrity == "ok":
            logger.info("  ✓ Integrity check passed")
        else:
            logger.warning(f"  ⚠ Integrity issues: {integrity}")
        
        # 6. Оптимизация (VACUUM)
        logger.info("Running VACUUM (this may take a while)...")
        start_time = time.time()
        conn.execute("VACUUM")
        vacuum_time = time.time() - start_time
        logger.info(f"  ✓ VACUUM completed in {vacuum_time:.2f} seconds")
        
        # 7. Финальная статистика
        cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
        new_size_bytes = cursor.fetchone()[0]
        new_size_mb = new_size_bytes / (1024 * 1024)
        saved_mb = size_mb - new_size_mb
        
        if saved_mb > 0:
            logger.info(f"  Space saved: {saved_mb:.2f} MB")
        
        conn.commit()
        logger.info(f"✅ {db_type} database optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error optimizing {db_type}: {e}")
        conn.rollback()
    finally:
        conn.close()


def main():
    """Основная функция"""
    logger.info("Starting database optimization...")
    
    # Пути к базам данных
    data_dir = Path("data")
    databases = {
        "metabolites": data_dir / "metabolites.db",
        "enzymes": data_dir / "enzymes.db",
        "proteins": data_dir / "proteins.db",
        "carbohydrates": data_dir / "carbohydrates.db",
        "lipids": data_dir / "lipids.db"
    }
    
    total_start = time.time()
    
    # Оптимизируем каждую БД
    for db_type, db_path in databases.items():
        optimize_database(db_path, db_type)
    
    total_time = time.time() - total_start
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 All databases optimized in {total_time:.2f} seconds!")
    logger.info(f"{'='*60}")
    
    # Рекомендации
    logger.info("\n📌 Recommendations:")
    logger.info("1. Run this script periodically (weekly/monthly)")
    logger.info("2. Consider running during low-traffic periods")
    logger.info("3. Monitor query performance after optimization")
    logger.info("4. Keep backups before major optimizations")


if __name__ == "__main__":
    main()
