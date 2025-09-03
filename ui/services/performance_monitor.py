"""
Мониторинг производительности запросов
"""
import time
import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Директория для логов производительности
PERF_LOG_DIR = Path("ui/logs")
PERF_LOG_DIR.mkdir(parents=True, exist_ok=True)


class PerformanceMonitor:
    """Мониторинг производительности приложения"""

    def __init__(self):
        self.query_stats = defaultdict(list)
        self.request_stats = []
        self.start_time = time.time()

    def log_query(self, db_type: str, query_type: str, duration: float,
                  records_count: int = 0, cache_hit: bool = False) -> None:
        """
        Логирует выполнение запроса

        Args:
            db_type: Тип базы данных
            query_type: Тип запроса (SELECT, COUNT, etc.)
            duration: Время выполнения в секундах
            records_count: Количество возвращенных записей
            cache_hit: Было ли попадание в кэш
        """
        stat = {
            'timestamp': datetime.now().isoformat(),
            'db_type': db_type,
            'query_type': query_type,
            'duration': duration,
            'records_count': records_count,
            'cache_hit': cache_hit
        }

        self.query_stats[db_type].append(stat)

        # Логируем медленные запросы
        if duration > 1.0:  # Более 1 секунды
            logger.warning(f"Slow query detected: {db_type} {query_type} took {duration:.2f}s")
        elif duration > 0.1:  # Более 100мс
            logger.info(f"Query performance: {db_type} {query_type} took {duration:.2f}s")

    def log_request(self, endpoint: str, method: str, duration: float,
                   status_code: int = 200) -> None:
        """
        Логирует HTTP запрос

        Args:
            endpoint: Конечная точка
            method: HTTP метод
            duration: Время выполнения
            status_code: Код ответа
        """
        stat = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'duration': duration,
            'status_code': status_code
        }

        self.request_stats.append(stat)

    def get_query_stats(self, db_type: Optional[str] = None,
                       hours: int = 24) -> Dict[str, Any]:
        """
        Получает статистику запросов

        Args:
            db_type: Тип базы данных (если None, то все)
            hours: За сколько часов собрать статистику

        Returns:
            Статистика запросов
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if db_type:
            stats = self.query_stats.get(db_type, [])
        else:
            stats = []
            for db_stats in self.query_stats.values():
                stats.extend(db_stats)

        # Фильтруем по времени
        recent_stats = [
            stat for stat in stats
            if datetime.fromisoformat(stat['timestamp']) > cutoff_time
        ]

        if not recent_stats:
            return {
                'total_queries': 0,
                'avg_duration': 0,
                'cache_hit_rate': 0,
                'slow_queries': 0
            }

        total_queries = len(recent_stats)
        total_duration = sum(stat['duration'] for stat in recent_stats)
        cache_hits = sum(1 for stat in recent_stats if stat['cache_hit'])
        slow_queries = sum(1 for stat in recent_stats if stat['duration'] > 1.0)

        return {
            'total_queries': total_queries,
            'avg_duration': total_duration / total_queries if total_queries > 0 else 0,
            'cache_hit_rate': cache_hits / total_queries if total_queries > 0 else 0,
            'slow_queries': slow_queries,
            'total_records': sum(stat['records_count'] for stat in recent_stats)
        }

    def get_performance_report(self) -> str:
        """
        Генерирует отчет о производительности

        Returns:
            Текстовый отчет
        """
        uptime = time.time() - self.start_time

        report = f"""
📊 Отчет о производительности
{'='*40}
⏱️  Время работы: {uptime:.1f} секунд ({uptime/3600:.1f} часов)

🔍 Статистика запросов:
"""

        for db_type in self.query_stats.keys():
            stats = self.get_query_stats(db_type, hours=24)
            if stats['total_queries'] > 0:
                report += f"""
📁 {db_type.upper()}:
  • Всего запросов: {stats['total_queries']}
  • Среднее время: {stats['avg_duration']*1000:.1f} мс
  • Попаданий в кэш: {stats['cache_hit_rate']*100:.1f}%
  • Медленных запросов: {stats['slow_queries']}
  • Всего записей: {stats['total_records']}
"""

        report += "\n💾 Статистика кэша:\n"
        try:
            from .cache_manager import cache_manager
            cache_stats = cache_manager.get_stats()
            report += f"""  • Всего файлов: {cache_stats['total_files']}
  • Валидных: {cache_stats['valid_files']}
  • Истекших: {cache_stats['expired_files']}
  • Размер: {cache_stats['total_size_mb']:.2f} MB
"""
        except ImportError:
            report += "  • Кэш недоступен\n"

        return report

    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Сохраняет отчет в файл

        Args:
            filename: Имя файла (если None, генерируется автоматически)

        Returns:
            Путь к сохраненному файлу
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.txt"

        report_path = PERF_LOG_DIR / filename

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self.get_performance_report())

            logger.info(f"Performance report saved: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return ""

    def reset_stats(self) -> None:
        """Сбрасывает статистику производительности"""
        self.query_stats.clear()
        self.request_stats.clear()
        self.start_time = time.time()
        logger.info("Performance statistics reset")


# Глобальный экземпляр монитора производительности
performance_monitor = PerformanceMonitor()


def profile_query(func):
    """
    Декоратор для профилирования запросов

    Args:
        func: Функция для профилирования

    Returns:
        Декорированная функция
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Определяем тип базы данных из аргументов
            if args and len(args) > 1:
                db_type = args[1] if isinstance(args[1], str) else "unknown"
            else:
                db_type = "unknown"

            # Определяем тип запроса
            query = args[2] if len(args) > 2 else ""
            query_type = query.split()[0].upper() if query else "UNKNOWN"

            # Определяем количество записей
            records_count = 0
            if isinstance(result, list):
                records_count = len(result)
            elif isinstance(result, int):
                records_count = result

            performance_monitor.log_query(
                db_type=db_type,
                query_type=query_type,
                duration=duration,
                records_count=records_count,
                cache_hit=False  # Будет установлено в cache_manager
            )

            return result

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Query profiling error: {e} (duration: {duration:.3f}s)")
            raise

    return wrapper
