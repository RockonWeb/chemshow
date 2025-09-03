"""
Конфигурационные настройки для приложения Справочник соединений
"""
import os
from pathlib import Path
from typing import Dict, Any

# Определяем корневую директорию проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# -------------------------
# Пути к базам данных
# -------------------------
DATABASE_PATHS = {
    "metabolites": os.getenv("METABOLITES_DB_PATH", str(DATA_DIR / "metabolites.db")),
    "enzymes": os.getenv("ENZYMES_DB_PATH", str(DATA_DIR / "enzymes.db")),
    "proteins": os.getenv("PROTEINS_DB_PATH", str(DATA_DIR / "proteins.db")),
    "carbohydrates": os.getenv("CARBOHYDRATES_DB_PATH", str(DATA_DIR / "carbohydrates.db")),
    "lipids": os.getenv("LIPIDS_DB_PATH", str(DATA_DIR / "lipids.db")),
}

# -------------------------
# Настройки поиска
# -------------------------
SEARCH_CONFIG = {
    "default_page_size": 50,
    "max_page_size": 200,
    "min_page_size": 25,
    "default_tolerance_ppm": 1000,
    "max_tolerance_ppm": 10000,
    "min_tolerance_ppm": 250,
}

# -------------------------
# Настройки UI
# -------------------------
UI_CONFIG = {
    "page_title": "Справочник соединений",
    "page_icon": "🧬",
    "layout": "centered",
    "cards_per_row": 3,
    "description_max_words": 6,
    "default_page_size": 50,  # Добавлено для совместимости
}

# -------------------------
# Внешние ссылки
# -------------------------
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

# -------------------------
# Настройки логирования
# -------------------------
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# -------------------------
# Типы организмов
# -------------------------
ORGANISM_TYPES = [
    "Все",
    "plant",
    "animal",
    "microorganism",
    "universal"
]

# -------------------------
# Пресеты поиска
# -------------------------
SEARCH_PRESETS = {
    "glucose": "Глюкоза",
    "dehydrogenase": "Dehydrogenase",
    "formaldehyde": "Formaldehyde",
    "atp": "ATP",
}

# -------------------------
# Вспомогательные функции
# -------------------------
def get_database_paths() -> Dict[str, str]:
    """Возвращает пути к базам данных с проверкой существования"""
    missing_dbs = []
    existing_dbs = []
    
    print(f"Checking databases in: {DATA_DIR}")
    
    for db_type, path in DATABASE_PATHS.items():
        if os.path.exists(path):
            existing_dbs.append(f"{db_type}: {path}")
        else:
            missing_dbs.append(f"{db_type}: {path}")

    if existing_dbs:
        print(f"Found databases: {len(existing_dbs)}")
        for db in existing_dbs:
            print(f"  ✓ {db}")
    
    if missing_dbs:
        print(f"Missing databases: {len(missing_dbs)}")
        for db in missing_dbs:
            print(f"  ✗ {db}")
        
        # Проверяем альтернативные пути
        alt_data_dir = DATA_DIR / "data"
        if alt_data_dir.exists():
            print(f"Found alternative data directory: {alt_data_dir}")
            for db_type in missing_dbs:
                db_name = db_type.split(':')[0] + ".db"
                alt_path = alt_data_dir / db_name
                if alt_path.exists():
                    print(f"  Alternative found: {alt_path}")

        raise FileNotFoundError(
            f"Следующие файлы баз данных не найдены: {', '.join(db.split(': ')[1] for db in missing_dbs)}"
        )

    return DATABASE_PATHS

def validate_search_params(query: str = None, mass: float = None, tolerance_ppm: int = None) -> Dict[str, Any]:
    """Валидация параметров поиска"""
    errors = []

    if query and len(query.strip()) < 2:
        errors.append("Запрос должен содержать минимум 2 символа")

    if mass and (not isinstance(mass, (int, float)) or mass <= 0 or mass > 1000000):
        errors.append("Масса должна быть положительным числом до 1,000,000")

    if tolerance_ppm:
        if tolerance_ppm < SEARCH_CONFIG["min_tolerance_ppm"]:
            errors.append(f"Допуск PPM должен быть не менее {SEARCH_CONFIG['min_tolerance_ppm']}")
        if tolerance_ppm > SEARCH_CONFIG["max_tolerance_ppm"]:
            errors.append(f"Допуск PPM должен быть не более {SEARCH_CONFIG['max_tolerance_ppm']}")

    return {"valid": len(errors) == 0, "errors": errors}
