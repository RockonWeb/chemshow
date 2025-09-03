"""
Вспомогательные функции для обработки данных и форматирования
"""
import math
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Внешние ссылки для различных баз данных
EXTERNAL_LINKS = {
    "hmdb_base": "https://hmdb.ca/metabolites/",
    "kegg_base": "https://www.kegg.jp/entry/",
    "chebi_base": "https://www.ebi.ac.uk/chebi/searchId.do?chebiId=",
    "pubchem_base": "https://pubchem.ncbi.nlm.nih.gov/compound/",
    "uniprot_base": "https://www.uniprot.org/uniprot/",
    "pdb_base": "https://www.rcsb.org/structure/",
    "ncbi_gene_base": "https://www.ncbi.nlm.nih.gov/gene/?term=",
    "expasy_base": "https://enzyme.expasy.org/EC/",
}


def truncate_description(text: str, max_words: int = 6) -> str:
    """Обрезает описание до указанного количества слов"""
    if not text or text == 'None':
        return text

    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = ' '.join(words[:max_words])
    return truncated + '...'


def format_chemical_formula(formula: str) -> str:
    """Преобразует химическую формулу в HTML с правильными индексами"""
    if not formula or formula == "—" or formula == "None":
        return formula

    import re

    # Заменяем цифры на подстрочные индексы
    formula = re.sub(r'(\d+)\+', r'<span class="superscript">\1+</span>', formula)

    # Обрабатываем обычные индексы (например, H2O, CO2)
    formula = re.sub(r'(\d+)', r'<span class="subscript">\1</span>', formula)

    # Обрабатываем отрицательные заряды (например, SO4^2-)
    formula = re.sub(r'\^(\d+)-', r'<span class="superscript">\1-</span>', formula)

    return f'<span class="formula">{formula}</span>'


def safe_get_value(data: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Безопасное получение значения из словаря с заменой 'None' на default"""
    value = data.get(key, default)
    if value == "None" or value is None:
        return default
    return value


def format_mass(mass: Any, unit: str = "Da") -> str:
    """Форматирует массу с правильными единицами измерения"""
    if isinstance(mass, (int, float)):
        if unit == "kDa":
            return f"{mass:.1f} kDa"
        else:
            return f"{mass:.6f} {unit}"
    return "—" if mass is None else str(mass)


def create_external_links(entity_type: str, entity_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Создает список внешних ссылок для сущности"""
    links = []

    if entity_type == "metabolite":
        if entity_data.get("hmdb_id"):
            links.append({
                "name": "HMDB",
                "url": f"{EXTERNAL_LINKS['hmdb_base']}{entity_data['hmdb_id']}",
                "icon": "🧬"
            })
        if entity_data.get("kegg_id"):
            links.append({
                "name": "KEGG",
                "url": f"{EXTERNAL_LINKS['kegg_base']}{entity_data['kegg_id']}",
                "icon": "🔗"
            })
        if entity_data.get("chebi_id"):
            links.append({
                "name": "ChEBI",
                "url": f"{EXTERNAL_LINKS['chebi_base']}{entity_data['chebi_id']}",
                "icon": "⚗️"
            })
        if entity_data.get("pubchem_cid"):
            links.append({
                "name": "PubChem",
                "url": f"{EXTERNAL_LINKS['pubchem_base']}{entity_data['pubchem_cid']}",
                "icon": "🧪"
            })

    elif entity_type == "enzyme":
        if entity_data.get("uniprot_id"):
            links.append({
                "name": "UniProt",
                "url": f"{EXTERNAL_LINKS['uniprot_base']}{entity_data['uniprot_id']}",
                "icon": "🧬"
            })
        if entity_data.get("kegg_enzyme_id"):
            links.append({
                "name": "KEGG",
                "url": f"{EXTERNAL_LINKS['kegg_base']}{entity_data['kegg_enzyme_id']}",
                "icon": "🔗"
            })
        if entity_data.get("ec_number"):
            links.append({
                "name": "ExPASy",
                "url": f"{EXTERNAL_LINKS['expasy_base']}{entity_data['ec_number']}",
                "icon": "⚗️"
            })

    elif entity_type == "protein":
        if entity_data.get("uniprot_id"):
            links.append({
                "name": "UniProt",
                "url": f"{EXTERNAL_LINKS['uniprot_base']}{entity_data['uniprot_id']}",
                "icon": "🧬"
            })
        if entity_data.get("pdb_id"):
            links.append({
                "name": "PDB",
                "url": f"{EXTERNAL_LINKS['pdb_base']}{entity_data['pdb_id']}",
                "icon": "🏗️"
            })
        if entity_data.get("gene_name"):
            links.append({
                "name": "NCBI Gene",
                "url": f"{EXTERNAL_LINKS['ncbi_gene_base']}{entity_data['gene_name']}",
                "icon": "🧬"
            })

    return links


def create_pills_list(values: List[str], max_items: int = 5) -> str:
    """Создает HTML-список pills из значений"""
    if not values:
        return ""

    pills = []
    for value in values[:max_items]:
        if value and value != "—" and value != "None":
            pills.append(f'<span class="pill">{value}</span>')

    if len(values) > max_items:
        pills.append(f'<span class="pill">+{len(values) - max_items}</span>')

    return " ".join(pills)


def calculate_pagination_info(total_items: int, page: int, page_size: int) -> Dict[str, Any]:
    """Рассчитывает информацию о пагинации"""
    total_pages = math.ceil(total_items / page_size) if total_items > 0 else 1

    return {
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": min(page, total_pages),
        "page_size": page_size,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "start_item": (page - 1) * page_size + 1 if total_items > 0 else 0,
        "end_item": min(page * page_size, total_items)
    }


def format_search_query(query: str) -> Optional[str]:
    """Форматирует поисковый запрос"""
    if not query or not query.strip():
        return None

    # Приводим к формату с заглавной буквы
    return query.strip().capitalize()


def validate_search_params(query: Optional[str] = None, mass: Optional[float] = None,
                          tolerance_ppm: int = 1000) -> Dict[str, Any]:
    """Валидирует параметры поиска"""
    # Конфигурация поиска
    SEARCH_CONFIG = {
        "default_page_size": 50,
        "max_page_size": 200,
        "min_page_size": 25,
        "default_tolerance_ppm": 1000,
        "max_tolerance_ppm": 10000,
        "min_tolerance_ppm": 250,
    }

    errors = []
    warnings = []

    # Валидация запроса
    if query and len(query.strip()) < 2:
        errors.append("Поисковый запрос должен содержать минимум 2 символа")

    # Валидация массы
    if mass is not None:
        if not isinstance(mass, (int, float)) or mass <= 0:
            errors.append("Масса должна быть положительным числом")
        elif mass > 1000000:
            warnings.append("Масса больше 1,000,000 Da - это может быть необычно")

    # Валидация допуска
    if tolerance_ppm < SEARCH_CONFIG["min_tolerance_ppm"]:
        errors.append(f"Допуск PPM должен быть не менее {SEARCH_CONFIG['min_tolerance_ppm']}")
    elif tolerance_ppm > SEARCH_CONFIG["max_tolerance_ppm"]:
        errors.append(f"Допуск PPM должен быть не более {SEARCH_CONFIG['max_tolerance_ppm']}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


def get_display_name(entity: Dict[str, Any], default: str = "Без названия") -> str:
    """Получает отображаемое имя с приоритетом на русский язык"""
    return safe_get_value(entity, "name_ru") or safe_get_value(entity, "name") or default


def create_metric_html(label: str, value: Any, unit: str = "") -> str:
    """Создает HTML для метрики"""
    formatted_value = f"{value} {unit}" if unit else str(value)
    if value is None or value == "—" or str(value).lower() == "none":
        formatted_value = "Не указано"

    return f"""
    <div class="metric-card">
        <div class="metric-value">{formatted_value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def create_stats_html(stats: Dict[str, Any]) -> str:
    """Создает HTML для статистики базы данных"""
    status_color = "#10B981" if stats.get("db_status") == "healthy" else "#EF4444"
    status_text = "OK" if stats.get("db_status") == "healthy" else "Нет файла"

    return f"""
    <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin: 1rem 0; flex-wrap: wrap;">
        <div class="stats-card">
            <div class="stats-title">Метаболиты</div>
            <div class="stats-value">{stats.get("metabolites", "—")}</div>
        </div>
        <div class="stats-card">
            <div class="stats-title">Ферменты</div>
            <div class="stats-value">{stats.get("enzymes", "—")}</div>
        </div>
        <div class="stats-card">
            <div class="stats-title">Белки</div>
            <div class="stats-value">{stats.get("proteins", "—")}</div>
        </div>
        <div class="stats-card">
            <div class="stats-title">Углеводы</div>
            <div class="stats-value">{stats.get("carbohydrates", "—")}</div>
        </div>
        <div class="stats-card">
            <div class="stats-title">Липиды</div>
            <div class="stats-value">{stats.get("lipids", "—")}</div>
        </div>
        <div class="stats-card">
            <div class="stats-title">Статус БД</div>
            <div class="stats-value" style="color: {status_color};">{status_text}</div>
        </div>
    </div>
    """
