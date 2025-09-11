"""
Система мониторинга и оптимизации производительности
"""
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

import psutil
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Метрики производительности"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    disk_io_read_bytes: int
    disk_io_write_bytes: int
    query_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    response_time_ms: float = 0.0
    active_connections: int = 0


@dataclass
class QueryStats:
    """Статистика запросов"""
    query_type: str
    db_type: str
    duration_ms: float
    cache_hit: bool
    timestamp: float
    records_count: int = 0


class PerformanceMonitor:
    """Монитор производительности приложения"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.query_history = deque(maxlen=max_history)
        self.slow_queries = deque(maxlen=100)
        self.query_stats = defaultdict(lambda: {"count": 0, "total_time": 0})
        self.lock = Lock()
        self.start_time = time.time()
        
        # Пороги для предупреждений
        self.thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "query_time_ms": 1000,
            "cache_hit_rate": 0.7
        }
    
    def collect_system_metrics(self) -> PerformanceMetrics:
        """Собирает системные метрики"""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            disk = psutil.disk_io_counters()
            
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu,
                memory_percent=memory,
                disk_io_read_bytes=disk.read_bytes if disk else 0,
                disk_io_write_bytes=disk.write_bytes if disk else 0
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=0,
                memory_percent=0,
                disk_io_read_bytes=0,
                disk_io_write_bytes=0
            )
    
    def log_query(
        self,
        db_type: str,
        query_type: str,
        duration: float,
        records_count: int = 0,
        cache_hit: bool = False
    ):
        """Логирует информацию о запросе"""
        with self.lock:
            duration_ms = duration * 1000
            
            # Создаем статистику запроса
            stats = QueryStats(
                query_type=query_type,
                db_type=db_type,
                duration_ms=duration_ms,
                cache_hit=cache_hit,
                timestamp=time.time(),
                records_count=records_count
            )
            
            # Добавляем в историю
            self.query_history.append(stats)
            
            # Обновляем агрегированную статистику
            key = f"{db_type}:{query_type}"
            self.query_stats[key]["count"] += 1
            self.query_stats[key]["total_time"] += duration_ms
            
            # Проверяем на медленный запрос
            if duration_ms > self.thresholds["query_time_ms"]:
                self.slow_queries.append(stats)
                logger.warning(f"Slow query detected: {db_type}:{query_type} took {duration_ms:.2f}ms")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Получает текущие метрики"""
        with self.lock:
            # Системные метрики
            sys_metrics = self.collect_system_metrics()
            
            # Статистика кэша
            cache_total = sum(1 for q in self.query_history if q.timestamp > time.time() - 60)
            cache_hits = sum(1 for q in self.query_history if q.cache_hit and q.timestamp > time.time() - 60)
            cache_hit_rate = (cache_hits / cache_total * 100) if cache_total > 0 else 0
            
            # Средняя скорость ответа
            recent_queries = [q for q in self.query_history if q.timestamp > time.time() - 60]
            avg_response_time = (
                sum(q.duration_ms for q in recent_queries) / len(recent_queries)
                if recent_queries else 0
            )
            
            return {
                "system": {
                    "cpu_percent": sys_metrics.cpu_percent,
                    "memory_percent": sys_metrics.memory_percent,
                    "disk_read_mb": sys_metrics.disk_io_read_bytes / (1024 * 1024),
                    "disk_write_mb": sys_metrics.disk_io_write_bytes / (1024 * 1024)
                },
                "queries": {
                    "total_last_minute": cache_total,
                    "cache_hit_rate": f"{cache_hit_rate:.1f}%",
                    "avg_response_time_ms": f"{avg_response_time:.2f}",
                    "slow_queries_count": len(self.slow_queries)
                },
                "uptime_hours": (time.time() - self.start_time) / 3600
            }
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Получает статистику запросов"""
        with self.lock:
            stats = {}
            
            for key, data in self.query_stats.items():
                avg_time = data["total_time"] / data["count"] if data["count"] > 0 else 0
                stats[key] = {
                    "count": data["count"],
                    "avg_time_ms": f"{avg_time:.2f}",
                    "total_time_ms": f"{data['total_time']:.2f}"
                }
            
            return stats
    
    def get_optimization_suggestions(self) -> List[str]:
        """Генерирует предложения по оптимизации"""
        suggestions = []
        metrics = self.get_current_metrics()
        
        # Проверка CPU
        if metrics["system"]["cpu_percent"] > self.thresholds["cpu_percent"]:
            suggestions.append(f"⚠️ Высокая загрузка CPU ({metrics['system']['cpu_percent']}%). Рекомендуется оптимизировать тяжелые операции.")
        
        # Проверка памяти
        if metrics["system"]["memory_percent"] > self.thresholds["memory_percent"]:
            suggestions.append(f"⚠️ Высокое использование памяти ({metrics['system']['memory_percent']}%). Рекомендуется проверить утечки памяти.")
        
        # Проверка кэша
        cache_hit_rate = float(metrics["queries"]["cache_hit_rate"].rstrip('%'))
        if cache_hit_rate < self.thresholds["cache_hit_rate"] * 100:
            suggestions.append(f"💡 Низкий показатель попаданий в кэш ({cache_hit_rate}%). Увеличьте размер кэша или TTL.")
        
        # Проверка медленных запросов
        if metrics["queries"]["slow_queries_count"] > 10:
            suggestions.append(f"🐌 Обнаружено {metrics['queries']['slow_queries_count']} медленных запросов. Рекомендуется добавить индексы.")
        
        # Анализ топ медленных запросов
        if self.slow_queries:
            slow_by_type = defaultdict(list)
            for q in self.slow_queries:
                slow_by_type[f"{q.db_type}:{q.query_type}"].append(q.duration_ms)
            
            for query_type, times in slow_by_type.items():
                avg_time = sum(times) / len(times)
                if avg_time > 2000:
                    suggestions.append(f"🔥 Критически медленный запрос {query_type}: среднее время {avg_time:.0f}ms")
        
        if not suggestions:
            suggestions.append("✅ Производительность в норме")
        
        return suggestions
    
    def export_metrics(self, filepath: str = "performance_report.json"):
        """Экспортирует метрики в файл"""
        with self.lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": self.get_current_metrics(),
                "query_statistics": self.get_query_statistics(),
                "slow_queries": [
                    {
                        "db_type": q.db_type,
                        "query_type": q.query_type,
                        "duration_ms": q.duration_ms,
                        "timestamp": datetime.fromtimestamp(q.timestamp).isoformat()
                    }
                    for q in list(self.slow_queries)[-20:]  # Последние 20 медленных запросов
                ],
                "suggestions": self.get_optimization_suggestions()
            }
            
            Path(filepath).write_text(json.dumps(report, indent=2, ensure_ascii=False))
            logger.info(f"Performance report exported to {filepath}")
            
            return report


# Глобальный экземпляр монитора
performance_monitor = PerformanceMonitor()