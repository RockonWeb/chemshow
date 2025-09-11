"""
Система рекомендаций похожих соединений
AI-powered поиск аналогов и кластеризация
Улучшенная версия с оптимизацией производительности
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import sqlite3
import re
import math
import os
import hashlib
import time
from collections import defaultdict
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Улучшенные импорты с более надежной обработкой ошибок
def safe_import():
    """Безопасный импорт всех зависимостей"""
    imports = {}
    
    # Импорт утилит
    try:
        from .utils import get_display_name, safe_get_value, format_mass
        imports['utils'] = True
    except ImportError:
        try:
            from utils import get_display_name, safe_get_value, format_mass
            imports['utils'] = True
        except ImportError:
            logger.warning("Утилиты не найдены, используем заглушки")
            # Заглушки для функций
            def get_display_name(entity, max_words=None):
                return entity.get('name', 'Без названия')
            def safe_get_value(entity, key, default='—'):
                return entity.get(key, default)
            def format_mass(mass):
                return f"{mass:.2f}" if mass else "—"
            imports['utils'] = False
    
    # Импорт настроек
    try:
        from ..config.settings import DATABASE_PATHS
        imports['settings'] = True
    except ImportError:
        try:
            from config.settings import DATABASE_PATHS
            imports['settings'] = True
        except ImportError:
            logger.warning("Настройки не найдены, используем значения по умолчанию")
            DATABASE_PATHS = {
                "metabolites": "/workspace/data/metabolites.db",
                "enzymes": "/workspace/data/enzymes.db", 
                "proteins": "/workspace/data/proteins.db",
                "carbohydrates": "/workspace/data/carbohydrates.db",
                "lipids": "/workspace/data/lipids.db"
            }
            imports['settings'] = False
    
    # Импорт RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs, Descriptors
        from rdkit.Chem.Fingerprints import FingerprintMols
        imports['rdkit'] = True
        logger.info("RDKit доступен для расширенного анализа")
    except ImportError:
        logger.warning("RDKit не доступен - будут использованы базовые алгоритмы")
        imports['rdkit'] = False
    
    return imports

# Глобальные импорты
IMPORTS = safe_import()
RDKIT_AVAILABLE = IMPORTS['rdkit']

if RDKIT_AVAILABLE:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors


def timing_decorator(func):
    """Декоратор для измерения времени выполнения"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} выполнена за {end_time - start_time:.3f}с")
        return result
    return wrapper


def cache_key(*args, **kwargs):
    """Генерация ключа кэша из аргументов"""
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()


class AdvancedCache:
    """Улучшенная система кэширования с TTL и размером"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша"""
        if key in self.cache:
            # Проверяем TTL
            if time.time() - self.access_times[key] < self.ttl:
                return self.cache[key]
            else:
                # Удаляем устаревшую запись
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key: str, value: Any):
        """Сохранить значение в кэш"""
        # Очищаем кэш если он переполнен
        if len(self.cache) >= self.max_size:
            self._cleanup_cache()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def _cleanup_cache(self):
        """Очистка старых записей кэша"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
        
        # Если все еще переполнен, удаляем самые старые
        if len(self.cache) >= self.max_size:
            sorted_keys = sorted(
                self.access_times.keys(),
                key=lambda k: self.access_times[k]
            )
            keys_to_remove = sorted_keys[:len(sorted_keys)//4]  # Удаляем 25%
            for key in keys_to_remove:
                del self.cache[key]
                del self.access_times[key]


class RecommendationsEngine:
    """Улучшенный движок для поиска рекомендаций и похожих соединений"""

    def __init__(self):
        self.compound_cache = AdvancedCache(max_size=2000, ttl=1800)  # 30 минут
        self.similarity_cache = AdvancedCache(max_size=5000, ttl=3600)  # 1 час
        self.fingerprint_cache = AdvancedCache(max_size=1000, ttl=7200)  # 2 часа
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Конфигурация весов для различных типов схожести
        self.similarity_weights = {
            'name': 0.25,
            'formula': 0.30,
            'mass': 0.15,
            'structure': 0.20,
            'properties': 0.10
        }
        
        # Пороги для различных критериев
        self.thresholds = {
            'mass_tolerance_percent': 0.1,  # 10% от массы
            'min_text_similarity': 0.1,
            'min_structural_similarity': 0.3
        }

    @timing_decorator
    def _clean_smiles_data(self, compound: Dict[str, Any]) -> Dict[str, Any]:
        """Улучшенная очистка данных SMILES"""
        cleaned_compound = compound.copy()
        
        smiles = cleaned_compound.get('smiles', '')
        if not self._is_valid_smiles(smiles):
            cleaned_compound['smiles'] = None
            logger.debug(f"Очищен невалидный SMILES для {cleaned_compound.get('name', 'Unknown')}")
        else:
            # Нормализация SMILES
            cleaned_compound['smiles'] = self._normalize_smiles(smiles)
        
        return cleaned_compound

    def _normalize_smiles(self, smiles: str) -> str:
        """Нормализация SMILES строки"""
        if not RDKIT_AVAILABLE or not smiles:
            return smiles
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        except Exception as e:
            logger.debug(f"Ошибка нормализации SMILES: {e}")
        
        return smiles

    @timing_decorator
    def find_similar_compounds(self, target_compound: Dict[str, Any],
                              database_type: str, limit: int = 10,
                              compounds_list: Optional[List[Dict[str, Any]]] = None,
                              use_parallel: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """
        Улучшенный поиск похожих соединений с параллельной обработкой
        """
        try:
            # Получаем данные соединений
            if compounds_list is not None:
                compounds_data = compounds_list
            else:
                compounds_data = self._load_compounds_from_db(database_type)
                
            if not compounds_data:
                return []

            # Очищаем данные
            target_compound = self._clean_smiles_data(target_compound)
            compounds_data = [self._clean_smiles_data(comp) for comp in compounds_data]

            # Параллельный расчет схожести
            if use_parallel and len(compounds_data) > 50:
                similarities = self._calculate_similarities_parallel(
                    target_compound, compounds_data, database_type
                )
            else:
                similarities = self._calculate_similarities_sequential(
                    target_compound, compounds_data, database_type
                )

            # Сортируем и фильтруем результаты
            similarities = [(comp, score) for comp, score in similarities if score > 0]
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:limit]

        except Exception as e:
            logger.error(f"Ошибка поиска похожих соединений: {e}")
            return []

    def _load_compounds_from_db(self, database_type: str) -> List[Dict[str, Any]]:
        """Загрузка соединений из базы данных с кэшированием"""
        cache_key_str = f"compounds_{database_type}"
        cached_data = self.compound_cache.get(cache_key_str)
        
        if cached_data:
            return cached_data

        if database_type not in DATABASE_PATHS:
            return []

        db_path = DATABASE_PATHS[database_type]
        if not os.path.exists(db_path):
            return []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Получаем структуру таблицы
            cursor.execute(f"PRAGMA table_info({database_type})")
            columns = [row[1] for row in cursor.fetchall()]

            if not columns:
                return []

            # Загружаем данные
            cursor.execute(f"SELECT * FROM {database_type}")
            all_compounds = cursor.fetchall()
            conn.close()

            # Преобразуем в словари
            compounds_data = [dict(zip(columns, row)) for row in all_compounds]
            
            # Кэшируем результат
            self.compound_cache.set(cache_key_str, compounds_data)
            
            return compounds_data

        except Exception as e:
            logger.error(f"Ошибка загрузки из БД {database_type}: {e}")
            return []

    def _calculate_similarities_parallel(self, target_compound: Dict[str, Any],
                                       compounds_data: List[Dict[str, Any]],
                                       database_type: str) -> List[Tuple[Dict[str, Any], float]]:
        """Параллельный расчет схожести"""
        similarities = []
        
        # Разбиваем на батчи для параллельной обработки
        batch_size = max(10, len(compounds_data) // 4)
        batches = [compounds_data[i:i + batch_size] 
                  for i in range(0, len(compounds_data), batch_size)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_batch = {
                executor.submit(
                    self._calculate_batch_similarities,
                    target_compound, batch, database_type
                ): batch for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_similarities = future.result()
                    similarities.extend(batch_similarities)
                except Exception as e:
                    logger.error(f"Ошибка в батче: {e}")
        
        return similarities

    def _calculate_batch_similarities(self, target_compound: Dict[str, Any],
                                    batch: List[Dict[str, Any]],
                                    database_type: str) -> List[Tuple[Dict[str, Any], float]]:
        """Расчет схожести для батча соединений"""
        similarities = []
        
        for compound in batch:
            if compound.get('id') == target_compound.get('id'):
                continue
                
            similarity_score = self._calculate_similarity(target_compound, compound, database_type)
            similarities.append((compound, similarity_score))
        
        return similarities

    def _calculate_similarities_sequential(self, target_compound: Dict[str, Any],
                                         compounds_data: List[Dict[str, Any]],
                                         database_type: str) -> List[Tuple[Dict[str, Any], float]]:
        """Последовательный расчет схожести"""
        similarities = []
        
        for compound in compounds_data:
            if compound.get('id') == target_compound.get('id'):
                continue
                
            similarity_score = self._calculate_similarity(target_compound, compound, database_type)
            similarities.append((compound, similarity_score))
        
        return similarities

    @lru_cache(maxsize=1000)
    def _calculate_similarity(self, target_compound: Dict[str, Any],
                            compound: Dict[str, Any], database_type: str) -> float:
        """
        Улучшенный расчет схожести с кэшированием
        """
        # Создаем ключ кэша
        cache_key_str = cache_key(
            target_compound.get('id'), 
            compound.get('id'), 
            database_type
        )
        
        cached_similarity = self.similarity_cache.get(cache_key_str)
        if cached_similarity is not None:
            return cached_similarity

        similarity = 0.0
        weights = self.similarity_weights

        # 1. Схожесть по названию
        name_sim = self._text_similarity(
            target_compound.get('name', ''),
            compound.get('name', '')
        )
        similarity += name_sim * weights['name']

        # 2. Схожесть по формуле
        if database_type in ['metabolites', 'carbohydrates', 'lipids']:
            formula_sim = self._formula_similarity(
                target_compound.get('formula', ''),
                compound.get('formula', '')
            )
            similarity += formula_sim * weights['formula']

        # 3. Схожесть по массе
        mass_sim = self._mass_similarity(
            target_compound.get('exact_mass'),
            compound.get('exact_mass')
        )
        similarity += mass_sim * weights['mass']

        # 4. Структурная схожесть
        if RDKIT_AVAILABLE:
            target_smiles = target_compound.get('smiles')
            comp_smiles = compound.get('smiles')
            
            if target_smiles and comp_smiles and \
               self._is_valid_smiles(target_smiles) and self._is_valid_smiles(comp_smiles):
                struct_sim = self._structural_similarity(target_smiles, comp_smiles)
                similarity += struct_sim * weights['structure']

        # 5. Схожесть по свойствам
        if database_type == 'enzymes':
            prop_sim = self._enzyme_similarity(target_compound, compound)
        elif database_type == 'proteins':
            prop_sim = self._protein_similarity(target_compound, compound)
        else:
            prop_sim = 0.0
            
        similarity += prop_sim * weights['properties']

        # Нормализуем результат
        final_similarity = min(similarity, 1.0)
        
        # Кэшируем результат
        self.similarity_cache.set(cache_key_str, final_similarity)
        
        return final_similarity

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Улучшенный расчет текстового сходства"""
        if not text1 or not text2:
            return 0.0

        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0

        # Используем несколько метрик
        # 1. Jaccard similarity для слов
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        jaccard_sim = len(words1 & words2) / len(words1 | words2)
        
        # 2. Схожесть подстрок
        substring_sim = 0.0
        if len(text1) > 2 and len(text2) > 2:
            # Ищем общие подстроки длиной 3+
            substrings1 = {text1[i:i+3] for i in range(len(text1)-2)}
            substrings2 = {text2[i:i+3] for i in range(len(text2)-2)}
            if substrings1 and substrings2:
                substring_sim = len(substrings1 & substrings2) / len(substrings1 | substrings2)
        
        # Комбинируем метрики
        return 0.7 * jaccard_sim + 0.3 * substring_sim

    def _formula_similarity(self, formula1: str, formula2: str) -> float:
        """Улучшенный расчет сходства химических формул"""
        if not formula1 or not formula2:
            return 0.0

        elements1 = self._parse_formula(formula1)
        elements2 = self._parse_formula(formula2)

        if not elements1 or not elements2:
            return 0.0

        # Рассчитываем сходство по элементам
        all_elements = set(elements1.keys()) | set(elements2.keys())
        
        if not all_elements:
            return 0.0

        # Векторное представление формул
        vector1 = [elements1.get(elem, 0) for elem in all_elements]
        vector2 = [elements2.get(elem, 0) for elem in all_elements]
        
        # Косинусное сходство
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
            
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        # Jaccard similarity для элементов
        set1 = set(elements1.keys())
        set2 = set(elements2.keys())
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0
        
        return 0.6 * cosine_sim + 0.4 * jaccard_sim

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """Улучшенный парсинг химической формулы"""
        if not formula:
            return {}
            
        elements = {}
        # Расширенный паттерн для парсинга
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        for element, count in matches:
            count = int(count) if count else 1
            elements[element] = elements.get(element, 0) + count

        return elements

    def _mass_similarity(self, mass1: Optional[float], mass2: Optional[float]) -> float:
        """Улучшенный расчет сходства по молекулярной массе"""
        if mass1 is None or mass2 is None:
            return 0.0

        mass1, mass2 = float(mass1), float(mass2)
        
        if mass1 == mass2:
            return 1.0

        # Адаптивная толерантность в зависимости от размера молекулы
        avg_mass = (mass1 + mass2) / 2
        tolerance = avg_mass * self.thresholds['mass_tolerance_percent']
        
        diff = abs(mass1 - mass2)
        
        if diff <= tolerance:
            # Линейная функция в пределах толерантности
            return 1.0 - (diff / tolerance)
        else:
            # Экспоненциальное затухание за пределами толерантности
            sigma = tolerance * 2
            return math.exp(-(diff ** 2) / (2 * sigma ** 2))

    @lru_cache(maxsize=500)
    def _structural_similarity(self, smiles1: str, smiles2: str) -> float:
        """Улучшенный расчет структурного сходства с кэшированием"""
        if not RDKIT_AVAILABLE:
            return 0.0

        # Проверяем кэш fingerprints
        fp_cache_key1 = f"fp_{smiles1}"
        fp_cache_key2 = f"fp_{smiles2}"
        
        fp1 = self.fingerprint_cache.get(fp_cache_key1)
        fp2 = self.fingerprint_cache.get(fp_cache_key2)

        try:
            if fp1 is None:
                mol1 = Chem.MolFromSmiles(smiles1)
                if mol1 is None:
                    return 0.0
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
                self.fingerprint_cache.set(fp_cache_key1, fp1)

            if fp2 is None:
                mol2 = Chem.MolFromSmiles(smiles2)
                if mol2 is None:
                    return 0.0
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
                self.fingerprint_cache.set(fp_cache_key2, fp2)

            # Tanimoto similarity
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            return similarity

        except Exception as e:
            logger.error(f"Ошибка расчета структурного сходства: {e}")
            return 0.0

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Улучшенная проверка валидности SMILES"""
        if not smiles or not isinstance(smiles, str):
            return False
        
        smiles = smiles.strip()
        
        # Проверяем на недопустимые значения
        invalid_values = {'0', 'None', 'null', '', 'nan', 'NaN', 'NULL', 'n/a', 'N/A'}
        if smiles in invalid_values:
            return False
        
        # Минимальная длина
        if len(smiles) < 2:
            return False
        
        # Проверяем базовые химические символы
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}@+-=#$%:;,.')
        if not all(char in valid_chars for char in smiles):
            return False
        
        # Если доступен RDKit, проверяем через него
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False
        
        return True

    def _enzyme_similarity(self, enzyme1: Dict[str, Any], enzyme2: Dict[str, Any]) -> float:
        """Улучшенный расчет сходства ферментов"""
        similarity = 0.0

        # EC номер (иерархическое сравнение)
        ec1 = enzyme1.get('ec_number', '')
        ec2 = enzyme2.get('ec_number', '')

        if ec1 and ec2:
            ec_parts1 = ec1.split('.')
            ec_parts2 = ec2.split('.')
            
            # Весы для разных уровней EC классификации
            level_weights = [0.4, 0.3, 0.2, 0.1]
            ec_similarity = 0.0
            
            for i in range(min(len(ec_parts1), len(ec_parts2), 4)):
                if ec_parts1[i] == ec_parts2[i]:
                    ec_similarity += level_weights[i]
                else:
                    break
            
            similarity += ec_similarity * 0.6

        # Семейство
        family1 = enzyme1.get('family', '')
        family2 = enzyme2.get('family', '')
        if family1 and family2:
            family_sim = 1.0 if family1.lower() == family2.lower() else 0.0
            similarity += family_sim * 0.3

        # Функция (текстовое сходство)
        func1 = enzyme1.get('function', '')
        func2 = enzyme2.get('function', '')
        if func1 and func2:
            func_sim = self._text_similarity(func1, func2)
            similarity += func_sim * 0.1

        return similarity

    def _protein_similarity(self, protein1: Dict[str, Any], protein2: Dict[str, Any]) -> float:
        """Улучшенный расчет сходства белков"""
        similarity = 0.0

        # Функция
        func1 = protein1.get('function', '')
        func2 = protein2.get('function', '')
        if func1 and func2:
            func_sim = self._text_similarity(func1, func2)
            similarity += func_sim * 0.4

        # Семейство
        family1 = protein1.get('family', '')
        family2 = protein2.get('family', '')
        if family1 and family2:
            family_sim = 1.0 if family1.lower() == family2.lower() else 0.0
            similarity += family_sim * 0.3

        # Организм
        org1 = protein1.get('organism', '')
        org2 = protein2.get('organism', '')
        if org1 and org2:
            org_sim = 1.0 if org1.lower() == org2.lower() else 0.0
            similarity += org_sim * 0.2

        # Длина последовательности (если доступна)
        len1 = protein1.get('sequence_length')
        len2 = protein2.get('sequence_length')
        if len1 and len2:
            len_sim = self._mass_similarity(float(len1), float(len2))  # Используем ту же логику
            similarity += len_sim * 0.1

        return similarity

    def _apply_filters(self, compounds: List[Dict[str, Any]], 
                      mass_range: Tuple[float, float],
                      smiles_only: bool, 
                      keyword_filter: str,
                      formula_elements: List[str]) -> List[Dict[str, Any]]:
        """Улучшенное применение фильтров с логированием"""
        initial_count = len(compounds)
        filtered_compounds = []

        for compound in compounds:
            # Фильтр по массе
            mass = compound.get('exact_mass')
            if mass is not None:
                if mass < mass_range[0] or mass > mass_range[1]:
                    continue

            # Фильтр по SMILES
            if smiles_only:
                if not self._is_valid_smiles(compound.get('smiles', '')):
                    continue

            # Фильтр по ключевым словам
            if keyword_filter:
                keywords = [kw.strip().lower() for kw in keyword_filter.split(',') if kw.strip()]
                name = compound.get('name', '').lower()
                if not any(keyword in name for keyword in keywords):
                    continue

            # Фильтр по элементам в формуле
            if formula_elements:
                formula = compound.get('formula', '')
                if formula:
                    elements_in_formula = set(self._parse_formula(formula).keys())
                    if not all(elem in elements_in_formula for elem in formula_elements):
                        continue

            filtered_compounds.append(compound)

        logger.info(f"Фильтрация: {initial_count} → {len(filtered_compounds)} соединений")
        return filtered_compounds

    @timing_decorator
    def cluster_compounds(self, compounds: List[Dict[str, Any]],
                         database_type: str, n_clusters: int = 5) -> Dict[str, Any]:
        """Улучшенная кластеризация соединений"""
        if not compounds or len(compounds) < n_clusters:
            return {"error": "Недостаточно данных для кластеризации"}

        try:
            # Создаем признаки для кластеризации
            features = self._extract_features(compounds, database_type)
            
            if not features or len(features[0]) == 0:
                return {"error": "Не удалось извлечь признаки для кластеризации"}

            # Нормализация признаков
            features_array = np.array(features)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Определяем оптимальное количество кластеров
            optimal_clusters = min(n_clusters, len(compounds) // 2)
            
            # Кластеризация с улучшенными параметрами
            kmeans = KMeans(
                n_clusters=optimal_clusters, 
                random_state=42, 
                n_init=10,
                max_iter=300
            )
            clusters = kmeans.fit_predict(features_scaled)

            # Группировка результатов
            cluster_results = defaultdict(list)
            for i, cluster_id in enumerate(clusters):
                cluster_results[cluster_id].append(compounds[i])

            # Статистика кластеров
            cluster_stats = {}
            for cluster_id, cluster_compounds in cluster_results.items():
                cluster_stats[cluster_id] = {
                    'size': len(cluster_compounds),
                    'avg_mass': np.mean([c.get('exact_mass', 0) for c in cluster_compounds if c.get('exact_mass')]),
                    'common_elements': self._find_common_elements(cluster_compounds)
                }

            return {
                "clusters": dict(cluster_results),
                "n_clusters": optimal_clusters,
                "total_compounds": len(compounds),
                "cluster_stats": cluster_stats,
                "silhouette_score": self._calculate_silhouette_score(features_scaled, clusters)
            }

        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}")
            return {"error": str(e)}

    def _extract_features(self, compounds: List[Dict[str, Any]], 
                         database_type: str) -> List[List[float]]:
        """Улучшенное извлечение признаков для кластеризации"""
        features = []

        for compound in compounds:
            feature_vector = []

            # Базовые признаки
            mass = compound.get('exact_mass', 0)
            feature_vector.append(float(mass) if mass else 0.0)
            
            name_len = len(compound.get('name', ''))
            feature_vector.append(float(name_len))
            
            formula_len = len(compound.get('formula', ''))
            feature_vector.append(float(formula_len))

            # Признаки формулы
            formula = compound.get('formula', '')
            if formula:
                elements = self._parse_formula(formula)
                # Количество уникальных элементов
                feature_vector.append(float(len(elements)))
                # Общее количество атомов
                feature_vector.append(float(sum(elements.values())))
                # Отношение C к другим элементам
                c_count = elements.get('C', 0)
                total_atoms = sum(elements.values())
                c_ratio = c_count / total_atoms if total_atoms > 0 else 0
                feature_vector.append(c_ratio)
            else:
                feature_vector.extend([0.0, 0.0, 0.0])

            # Специфичные признаки по типу базы
            if database_type == 'enzymes':
                ec_number = compound.get('ec_number', '0.0.0.0')
                ec_parts = ec_number.split('.')
                for i in range(4):
                    if i < len(ec_parts):
                        try:
                            feature_vector.append(float(ec_parts[i]))
                        except:
                            feature_vector.append(0.0)
                    else:
                        feature_vector.append(0.0)
                        
            elif database_type == 'proteins':
                seq_len = compound.get('sequence_length', 0)
                feature_vector.append(float(seq_len) if seq_len else 0.0)
                
                func_len = len(compound.get('function', ''))
                feature_vector.append(float(func_len))
                
                family_len = len(compound.get('family', ''))
                feature_vector.append(float(family_len))
                
            else:
                # Для метаболитов, углеводов, липидов
                class_len = len(compound.get('class_name', ''))
                feature_vector.append(float(class_len))
                
                # SMILES признаки
                smiles = compound.get('smiles', '')
                if smiles and self._is_valid_smiles(smiles):
                    feature_vector.append(float(len(smiles)))
                    feature_vector.append(float(smiles.count('=')))  # Двойные связи
                    feature_vector.append(float(smiles.count('#')))  # Тройные связи
                    feature_vector.append(float(smiles.count('(')))  # Ветвления
                else:
                    feature_vector.extend([0.0, 0.0, 0.0, 0.0])

            features.append(feature_vector)

        return features

    def _find_common_elements(self, compounds: List[Dict[str, Any]]) -> List[str]:
        """Находит общие химические элементы в кластере"""
        if not compounds:
            return []
            
        element_sets = []
        for compound in compounds:
            formula = compound.get('formula', '')
            if formula:
                elements = set(self._parse_formula(formula).keys())
                element_sets.append(elements)
        
        if not element_sets:
            return []
            
        # Находим пересечение всех множеств
        common_elements = element_sets[0]
        for elem_set in element_sets[1:]:
            common_elements &= elem_set
            
        return list(common_elements)

    def _calculate_silhouette_score(self, features: np.ndarray, clusters: np.ndarray) -> float:
        """Расчет коэффициента силуэта для оценки качества кластеризации"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(clusters)) > 1:
                return silhouette_score(features, clusters)
        except:
            pass
        return 0.0

    def get_recommendation_explanation(self, target_compound: Dict[str, Any],
                                     similar_compound: Dict[str, Any],
                                     similarity_score: float) -> str:
        """Улучшенная генерация объяснения рекомендации"""
        explanations = []

        # Объяснение по названию
        name1 = target_compound.get('name', '').lower()
        name2 = similar_compound.get('name', '').lower()
        
        if name1 and name2:
            common_words = set(name1.split()) & set(name2.split())
            if common_words:
                explanations.append(f"Общие слова: {', '.join(sorted(common_words))}")

        # Объяснение по формуле
        formula1 = target_compound.get('formula', '')
        formula2 = similar_compound.get('formula', '')
        
        if formula1 and formula2:
            elements1 = set(self._parse_formula(formula1).keys())
            elements2 = set(self._parse_formula(formula2).keys())
            common_elements = elements1 & elements2
            
            if common_elements:
                explanations.append(f"Общие элементы: {', '.join(sorted(common_elements))}")

        # Объяснение по массе
        mass1 = target_compound.get('exact_mass')
        mass2 = similar_compound.get('exact_mass')
        
        if mass1 and mass2:
            mass_diff = abs(mass1 - mass2)
            mass_diff_percent = (mass_diff / max(mass1, mass2)) * 100
            
            if mass_diff_percent < 5:
                explanations.append(f"Очень близкие массы (разница {mass_diff_percent:.1f}%)")
            elif mass_diff_percent < 15:
                explanations.append(f"Близкие массы (разница {mass_diff_percent:.1f}%)")

        # Объяснение по структуре
        if RDKIT_AVAILABLE and similarity_score > 0.6:
            smiles1 = target_compound.get('smiles')
            smiles2 = similar_compound.get('smiles')
            if smiles1 and smiles2:
                struct_sim = self._structural_similarity(smiles1, smiles2)
                if struct_sim > 0.7:
                    explanations.append(f"Высокая структурная схожесть ({struct_sim:.1%})")

        if not explanations:
            explanations.append(f"Общий коэффициент схожести: {similarity_score:.1%}")

        return " | ".join(explanations)


def render_recommendations_interface():
    """Улучшенный интерфейс системы рекомендаций"""
    st.header("🎯 Система рекомендаций")
    
    # Инициализация движка
    if 'recommendation_engine' not in st.session_state:
        st.session_state.recommendation_engine = RecommendationsEngine()
    
    engine = st.session_state.recommendation_engine

    # Выбор типа базы данных
    database_options = {
        "metabolites": "🧬 Метаболиты",
        "enzymes": "🧪 Ферменты", 
        "proteins": "🔬 Белки",
        "carbohydrates": "🌾 Углеводы",
        "lipids": "🫧 Липиды"
    }

    selected_db = st.selectbox(
        "Выберите тип соединений для анализа:",
        options=list(database_options.keys()),
        format_func=lambda x: database_options[x]
    )

    if selected_db:
        # Загрузка данных с прогресс-баром
        with st.spinner("Загружаю данные..."):
            compounds_list = engine._load_compounds_from_db(selected_db)

        if not compounds_list:
            st.error(f"❌ Не удалось загрузить данные из базы {database_options[selected_db]}")
            return

        st.success(f"✅ Загружено {len(compounds_list):,} соединений из базы {database_options[selected_db]}")

        # Очистка данных
        with st.spinner("Обрабатываю данные..."):
            compounds_list = [engine._clean_smiles_data(comp) for comp in compounds_list]

        # Выбор целевого соединения с поиском
        st.subheader("🎯 Выбор целевого соединения")
        
        # Поиск по названию
        search_query = st.text_input(
            "Поиск соединения по названию:",
            placeholder="Введите часть названия для поиска..."
        )
        
        if search_query:
            # Фильтруем соединения по поисковому запросу
            filtered_compounds = [
                comp for comp in compounds_list 
                if search_query.lower() in comp.get('name', '').lower()
            ]
            if filtered_compounds:
                st.info(f"Найдено {len(filtered_compounds)} соединений по запросу '{search_query}'")
                display_compounds = filtered_compounds[:100]  # Ограничиваем для производительности
            else:
                st.warning(f"Не найдено соединений по запросу '{search_query}'")
                display_compounds = compounds_list[:100]
        else:
            display_compounds = compounds_list[:100]

        # Выбор соединения
        compound_names = [
            f"{c.get('name', 'Без названия')[:50]} (ID: {c.get('id', '—')}, "
            f"Масса: {c.get('exact_mass', '—')})" 
            for c in display_compounds
        ]
        
        selected_compound_idx = st.selectbox(
            "Выберите соединение для поиска аналогов:",
            options=range(len(display_compounds)),
            format_func=lambda x: compound_names[x]
        )

        target_compound = display_compounds[selected_compound_idx]

        # Отображение информации о выбранном соединении
        with st.expander("ℹ️ Информация о выбранном соединении", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Название:** {target_compound.get('name', '—')}")
                st.write(f"**Формула:** {target_compound.get('formula', '—')}")
                st.write(f"**Масса:** {target_compound.get('exact_mass', '—')} Da")
            with col2:
                st.write(f"**ID:** {target_compound.get('id', '—')}")
                st.write(f"**SMILES:** {target_compound.get('smiles', '—')}")
                if target_compound.get('class_name'):
                    st.write(f"**Класс:** {target_compound.get('class_name')}")

        # Параметры поиска
        st.subheader("⚙️ Параметры поиска")

        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("Количество рекомендаций:", 5, 100, 15)
            use_parallel = st.checkbox("Параллельная обработка", value=True, 
                                     help="Ускоряет поиск для больших баз данных")
        with col2:
            min_similarity = st.slider("Минимальная схожесть (%):", 0, 100, 20) / 100.0
            show_explanations = st.checkbox("Показать объяснения", value=True)

        # Продвинутые фильтры
        with st.expander("🔍 Продвинутые фильтры"):
            col3, col4 = st.columns(2)

            with col3:
                # Фильтр по массе
                mass_range = st.slider(
                    "Диапазон массы (Da):",
                    0.0, 2000.0, (0.0, 2000.0),
                    help="Ограничить поиск соединениями в указанном диапазоне масс"
                )

                # Фильтр по наличию SMILES
                smiles_only = st.checkbox(
                    "Только с валидными SMILES",
                    help="Показывать только соединения с валидными SMILES"
                )

            with col4:
                # Фильтр по ключевым словам
                keyword_filter = st.text_input(
                    "Ключевые слова в названии:",
                    placeholder="glucose, acid, dehydrogenase",
                    help="Фильтровать по словам в названии (через запятую)"
                )

                # Фильтр по элементам в формуле
                formula_elements = st.multiselect(
                    "Обязательные элементы:",
                    options=["C", "H", "O", "N", "P", "S", "Cl", "Br", "I", "F", "Na", "K", "Ca", "Mg"],
                    help="Соединения должны содержать выбранные элементы"
                )

        # Поиск рекомендаций
        if st.button("🔍 Найти похожие соединения", type="primary"):
            with st.spinner("Ищу похожие соединения..."):
                start_time = time.time()
                
                # Применяем фильтры
                filtered_compounds = engine._apply_filters(
                    compounds_list, mass_range, smiles_only, 
                    keyword_filter, formula_elements
                )

                if len(filtered_compounds) < 2:
                    st.warning("⚠️ После применения фильтров осталось слишком мало соединений")
                    return

                # Поиск рекомендаций
                similar_compounds = engine.find_similar_compounds(
                    target_compound, selected_db, limit, 
                    filtered_compounds, use_parallel
                )

                # Фильтруем по минимальной схожести
                final_results = [
                    (comp, score) for comp, score in similar_compounds 
                    if score >= min_similarity
                ]

                search_time = time.time() - start_time
                
                # Сохраняем результаты
                st.session_state.recommendation_results = final_results
                st.session_state.target_compound = target_compound
                st.session_state.search_params = {
                    'database': selected_db,
                    'limit': limit,
                    'min_similarity': min_similarity,
                    'search_time': search_time,
                    'show_explanations': show_explanations
                }

        # Отображение результатов
        if hasattr(st.session_state, 'recommendation_results') and st.session_state.recommendation_results:
            results = st.session_state.recommendation_results
            target = st.session_state.target_compound
            params = st.session_state.search_params

            st.success(f"🎯 Найдено {len(results)} похожих соединений за {params['search_time']:.2f}с")
            
            st.subheader(f"📊 Рекомендации для: {target.get('name', 'Без названия')}")

            if results:
                # Таблица результатов
                result_data = []
                for i, (comp, similarity) in enumerate(results):
                    row_data = {
                        "№": i + 1,
                        "Название": comp.get('name', '—'),
                        "Схожесть": f"{similarity:.1%}",
                        "Формула": comp.get('formula', '—'),
                        "Масса (Da)": f"{comp.get('exact_mass', 0):.2f}" if comp.get('exact_mass') else "—",
                    }
                    
                    if params.get('show_explanations', True):
                        row_data["Объяснение"] = engine.get_recommendation_explanation(
                            target, comp, similarity
                        )
                    
                    result_data.append(row_data)

                df = pd.DataFrame(result_data)
                
                # Настройка отображения таблицы
                column_config = {
                    "№": st.column_config.NumberColumn(width="small"),
                    "Название": st.column_config.TextColumn(width="large"),
                    "Схожесть": st.column_config.TextColumn(width="small"),
                    "Формула": st.column_config.TextColumn(width="medium"),
                    "Масса (Da)": st.column_config.NumberColumn(width="small", format="%.2f"),
                }
                
                if params.get('show_explanations', True):
                    column_config["Объяснение"] = st.column_config.TextColumn(width="large")

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )

                # Визуализация схожести
                st.subheader("📈 Визуализация результатов")
                
                similarities = [sim for _, sim in results]
                names = [comp.get('name', '—')[:30] + '...' if len(comp.get('name', '')) > 30 
                        else comp.get('name', '—') for comp, _ in results]

                # График схожести
                fig = go.Figure()
                
                # Цветовая схема на основе схожести
                colors = ['#2E8B57' if s >= 0.7 else '#4682B4' if s >= 0.5 else '#CD853F' 
                         for s in similarities]
                
                fig.add_trace(go.Bar(
                    x=names,
                    y=[s * 100 for s in similarities],
                    marker_color=colors,
                    text=[f'{s*100:.1f}%' for s in similarities],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Схожесть: %{y:.1f}%<extra></extra>'
                ))

                fig.update_layout(
                    title=f"Схожесть с соединением: {target.get('name', '—')}",
                    xaxis_title="Соединение",
                    yaxis_title="Схожесть (%)",
                    height=500,
                    xaxis_tickangle=-45,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Экспорт результатов
                st.subheader("💾 Экспорт результатов")
                
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = df.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="📥 Скачать результаты (CSV)",
                        data=csv_data,
                        file_name=f"recommendations_{selected_db}_{target.get('name', 'compound')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # JSON экспорт с полными данными
                    json_data = {
                        'target_compound': target,
                        'recommendations': [
                            {'compound': comp, 'similarity': float(sim)} 
                            for comp, sim in results
                        ],
                        'search_parameters': params
                    }
                    import json
                    json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Скачать данные (JSON)",
                        data=json_str,
                        file_name=f"recommendations_{selected_db}_{target.get('name', 'compound')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            else:
                st.info("Не найдено соединений, удовлетворяющих критериям поиска")

        # Кластеризация
        st.divider()
        st.subheader("📊 Кластерный анализ")

        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Количество кластеров:", 2, 15, 5)
        with col2:
            cluster_sample_size = st.slider("Размер выборки для кластеризации:", 100, min(2000, len(compounds_list)), 500)

        if st.button("🎯 Выполнить кластеризацию"):
            with st.spinner("Выполняю кластерный анализ..."):
                # Используем выборку для ускорения
                sample_compounds = compounds_list[:cluster_sample_size]
                cluster_results = engine.cluster_compounds(sample_compounds, selected_db, n_clusters)

                if "error" not in cluster_results:
                    st.success(f"✅ Создано {cluster_results['n_clusters']} кластеров из {cluster_results['total_compounds']} соединений")
                    
                    if 'silhouette_score' in cluster_results and cluster_results['silhouette_score'] > 0:
                        st.info(f"📊 Коэффициент силуэта: {cluster_results['silhouette_score']:.3f} (качество кластеризации)")

                    clusters = cluster_results['clusters']
                    cluster_stats = cluster_results.get('cluster_stats', {})

                    # Визуализация кластеров
                    st.subheader("📈 Визуализация кластеров")
                    
                    # График размеров кластеров
                    cluster_sizes = [len(compounds) for compounds in clusters.values()]
                    cluster_labels = [f"Кластер {i+1}" for i in range(len(clusters))]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=cluster_labels,
                        values=cluster_sizes,
                        hole=0.3
                    )])
                    fig_pie.update_layout(title="Распределение соединений по кластерам")
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Отображение кластеров
                    for cluster_id, cluster_compounds in clusters.items():
                        stats = cluster_stats.get(cluster_id, {})
                        
                        with st.expander(f"Кластер {cluster_id + 1} ({len(cluster_compounds)} соединений)"):
                            # Статистика кластера
                            if stats:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Размер кластера", len(cluster_compounds))
                                with col2:
                                    avg_mass = stats.get('avg_mass', 0)
                                    st.metric("Средняя масса", f"{avg_mass:.1f} Da" if avg_mass else "—")
                                with col3:
                                    common_elems = stats.get('common_elements', [])
                                    st.metric("Общие элементы", f"{len(common_elems)} элементов")
                                    if common_elems:
                                        st.write(f"Элементы: {', '.join(common_elems)}")

                            # Показываем первые 10 соединений кластера
                            cluster_data = []
                            for comp in cluster_compounds[:10]:
                                cluster_data.append({
                                    "Название": comp.get('name', '—'),
                                    "Формула": comp.get('formula', '—'),
                                    "Масса": f"{comp.get('exact_mass', 0):.2f}" if comp.get('exact_mass') else "—"
                                })

                            if cluster_data:
                                cluster_df = pd.DataFrame(cluster_data)
                                st.dataframe(cluster_df, use_container_width=True, hide_index=True)
                                
                                if len(cluster_compounds) > 10:
                                    st.info(f"Показаны первые 10 из {len(cluster_compounds)} соединений кластера")

                    # Экспорт кластеров
                    st.subheader("💾 Экспорт кластеров")
                    
                    export_data = []
                    for cluster_id, cluster_compounds in clusters.items():
                        for comp in cluster_compounds:
                            export_data.append({
                                "Кластер": f"Кластер {cluster_id + 1}",
                                "Название": comp.get('name', '—'),
                                "Формула": comp.get('formula', '—'),
                                "Масса": comp.get('exact_mass', '—'),
                                "Тип": database_options[selected_db]
                            })

                    if export_data:
                        export_df = pd.DataFrame(export_data)
                        csv_cluster_data = export_df.to_csv(index=False, encoding='utf-8')
                        st.download_button(
                            label="📥 Скачать кластеры (CSV)",
                            data=csv_cluster_data,
                            file_name=f"clusters_{selected_db}_{n_clusters}_clusters.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                else:
                    st.error(f"❌ Ошибка кластеризации: {cluster_results['error']}")

    # Информация о системе
    with st.expander("ℹ️ О системе рекомендаций"):
        st.markdown("""
        **🎯 Улучшенная система рекомендаций** использует комплексный подход:

        ### 🔍 Алгоритмы схожести:
        - **Текстовое сходство**: Jaccard similarity + анализ подстрок
        - **Химическая формула**: Векторное сравнение элементного состава
        - **Молекулярная масса**: Адаптивная толерантность + гауссово сходство
        - **Структурное сходство**: Morgan fingerprints + Tanimoto similarity (RDKit)
        - **Функциональные свойства**: Иерархическое сравнение для ферментов и белков

        ### ⚡ Оптимизации производительности:
        - **Многоуровневое кэширование** с TTL
        - **Параллельная обработка** для больших датасетов
        - **Умная нормализация** SMILES структур
        - **Батчевая обработка** схожести

        ### 📊 Кластерный анализ:
        - **K-means кластеризация** с оптимизированными признаками
        - **Коэффициент силуэта** для оценки качества
        - **Статистика кластеров** с общими элементами
        - **Визуализация** распределения

        ### 🛠️ Технические улучшения:
        - Надежная обработка импортов
        - Расширенное логирование и профилирование
        - Гибкая система фильтрации
        - Экспорт в CSV и JSON форматах
        """)

        # Статистика производительности
        if hasattr(st.session_state, 'recommendation_engine'):
            engine = st.session_state.recommendation_engine
            st.subheader("📈 Статистика кэша")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Кэш соединений", len(engine.compound_cache.cache))
            with col2:
                st.metric("Кэш схожести", len(engine.similarity_cache.cache))
            with col3:
                st.metric("Кэш fingerprints", len(engine.fingerprint_cache.cache))


if __name__ == "__main__":
    render_recommendations_interface()