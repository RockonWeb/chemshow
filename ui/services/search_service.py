"""
Сервис унифицированного поиска по всем базам данных
"""
import logging
from typing import Dict, Any, Optional, List
from .database import (
    metabolite_repo, enzyme_repo, protein_repo,
    carbohydrate_repo, lipid_repo
)

logger = logging.getLogger(__name__)


class SearchService:
    """Сервис для выполнения унифицированного поиска"""

    def __init__(self):
        self.repositories = {
            "metabolites": metabolite_repo,
            "enzymes": enzyme_repo,
            "proteins": protein_repo,
            "carbohydrates": carbohydrate_repo,
            "lipids": lipid_repo
        }

    def unified_search(self, query: str = None, mass: float = None,
                      tolerance_ppm: int = 1000, organism_type: str = "Все",
                      page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """
        Выполняет унифицированный поиск по всем базам данных

        Args:
            query: Поисковый запрос
            mass: Масса для поиска (Da для метаболитов, будет конвертирована для ферментов/белков)
            tolerance_ppm: Допуск в ppm
            organism_type: Тип организма для фильтрации
            page: Номер страницы
            page_size: Размер страницы

        Returns:
            Словарь с результатами поиска по каждой базе данных
        """
        # Конвертация типов для безопасности (обработка строковых значений)
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

        try:
            if isinstance(page, str):
                page = int(page) if page.strip() else 1
            elif page is not None:
                page = int(page)
        except (ValueError, TypeError):
            page = 1

        try:
            if isinstance(page_size, str):
                page_size = int(page_size) if page_size.strip() else 50
            elif page_size is not None:
                page_size = int(page_size)
        except (ValueError, TypeError):
            page_size = 50

        if query is None:
            query = ""

        # Логирование для отладки типов параметров
        logger.debug(f"Search parameters - query: {repr(query)} (type: {type(query)}), "
                    f"mass: {repr(mass)} (type: {type(mass)}), "
                    f"tolerance_ppm: {repr(tolerance_ppm)} (type: {type(tolerance_ppm)}), "
                    f"page: {repr(page)} (type: {type(page)}), "
                    f"page_size: {repr(page_size)} (type: {type(page_size)})")

        results = {
            "metabolites": {"data": [], "total": 0},
            "enzymes": {"data": [], "total": 0},
            "proteins": {"data": [], "total": 0},
            "carbohydrates": {"data": [], "total": 0},
            "lipids": {"data": [], "total": 0}
        }

        # Проверяем, что есть что искать
        if not query and not mass:
            logger.warning("Empty search query and mass")
            return results

        # Если есть только масса, очищаем запрос для поиска только по массе
        if mass and mass > 0 and not query:
            query = ""

        # Поиск по метаболитам
        if mass and mass > 0:
            logger.info(f"Searching metabolites by mass: {mass} ± {tolerance_ppm} ppm")
            try:
                results["metabolites"]["data"] = metabolite_repo.search(
                    mass=mass, tolerance_ppm=tolerance_ppm,
                    limit=page_size, offset=(page - 1) * page_size
                )
                results["metabolites"]["total"] = metabolite_repo.count_search(
                    mass=mass, tolerance_ppm=tolerance_ppm
                )
            except Exception as e:
                logger.error(f"Metabolite search error: {e}")
        elif query and query.strip():
            logger.info(f"Searching metabolites by query: {query}")
            try:
                results["metabolites"]["data"] = metabolite_repo.search(
                    query=query, limit=page_size, offset=(page - 1) * page_size
                )
                results["metabolites"]["total"] = metabolite_repo.count_search(query=query)
            except Exception as e:
                logger.error(f"Metabolite search error: {e}")

        # Поиск по ферментам (по тексту и/или массе)
        if (query and query.strip()) or mass:
            try:
                results["enzymes"]["data"] = enzyme_repo.search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm,
                    organism_type=organism_type, limit=page_size,
                    offset=(page - 1) * page_size
                )
                results["enzymes"]["total"] = enzyme_repo.count_search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm,
                    organism_type=organism_type
                )
            except Exception as e:
                logger.error(f"Enzyme search error: {e}")

        # Поиск по белкам (по тексту и/или массе)
        if (query and query.strip()) or mass:
            try:
                results["proteins"]["data"] = protein_repo.search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm,
                    organism_type=organism_type, limit=page_size,
                    offset=(page - 1) * page_size
                )
                results["proteins"]["total"] = protein_repo.count_search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm,
                    organism_type=organism_type
                )
            except Exception as e:
                logger.error(f"Protein search error: {e}")

        # Поиск по углеводам (по тексту и/или массе)
        if (query and query.strip()) or mass:
            try:
                results["carbohydrates"]["data"] = carbohydrate_repo.search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm,
                    limit=page_size, offset=(page - 1) * page_size
                )
                results["carbohydrates"]["total"] = carbohydrate_repo.count_search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm
                )
            except Exception as e:
                logger.error(f"Carbohydrate search error: {e}")

        # Поиск по липидам (по тексту и/или массе)
        if (query and query.strip()) or mass:
            try:
                results["lipids"]["data"] = lipid_repo.search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm,
                    limit=page_size, offset=(page - 1) * page_size
                )
                results["lipids"]["total"] = lipid_repo.count_search(
                    query=query, mass=mass, tolerance_ppm=tolerance_ppm
                )
            except Exception as e:
                logger.error(f"Lipid search error: {e}")

        return results

    def get_search_totals(self, search_results: Dict[str, Any]) -> Dict[str, int]:
        """Извлекает итоговые количества из результатов поиска"""
        return {
            "metabolites": search_results.get("metabolites", {}).get("total", 0),
            "enzymes": search_results.get("enzymes", {}).get("total", 0),
            "proteins": search_results.get("proteins", {}).get("total", 0),
            "carbohydrates": search_results.get("carbohydrates", {}).get("total", 0),
            "lipids": search_results.get("lipids", {}).get("total", 0)
        }

    def get_max_results_for_pagination(self, search_totals: Dict[str, int]) -> int:
        """Определяет максимальное количество результатов для пагинации"""
        return max(search_totals.values())

    def has_results(self, search_results: Dict[str, Any]) -> bool:
        """Проверяет, есть ли результаты поиска"""
        totals = self.get_search_totals(search_results)
        return sum(totals.values()) > 0


# Глобальный экземпляр сервиса поиска
search_service = SearchService()
