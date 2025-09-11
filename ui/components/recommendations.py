"""
Система рекомендаций похожих соединений
AI-powered поиск аналогов и кластеризация
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Optional, Tuple
import logging
import sqlite3
import re
import math
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

# Импорт вспомогательных функций
try:
    # Try absolute import first
    from components.utils import get_display_name, safe_get_value, format_mass
except ImportError:
    # Fallback to relative import
    try:
        from .utils import get_display_name, safe_get_value, format_mass
    except ImportError:
        from utils import get_display_name, safe_get_value, format_mass

# Проверяем доступность RDKit для продвинутых расчетов
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs, Descriptors
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
    logger.info("RDKit доступен для рекомендаций")
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit не доступен - базовые рекомендации будут ограничены")


class RecommendationsEngine:
    """Движок для поиска рекомендаций и похожих соединений"""

    def __init__(self):
        self.compound_cache = {}
        self.similarity_cache = {}

    def _clean_smiles_data(self, compound: Dict[str, Any]) -> Dict[str, Any]:
        """Очищает данные SMILES от невалидных значений"""
        cleaned_compound = compound.copy()
        
        # Очищаем SMILES поле
        if 'smiles' in cleaned_compound:
            smiles = cleaned_compound['smiles']
            if not self._is_valid_smiles(smiles):
                cleaned_compound['smiles'] = None
                logger.debug(f"Очищен невалидный SMILES для соединения {cleaned_compound.get('name', 'Unknown')}")
        
        return cleaned_compound

    def find_similar_compounds(self, target_compound: Dict[str, Any],
                              database_type: str, limit: int = 10,
                              compounds_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Поиск похожих соединений на основе различных критериев
        """
        try:
            if compounds_list is not None:
                # Используем переданный список соединений
                compounds_data = compounds_list
            else:
                # Загружаем данные из базы данных
                # Try absolute import first
                try:
                    from config.settings import DATABASE_PATHS
                except ImportError:
                    # Fallback to relative import
                    from ..config.settings import DATABASE_PATHS

                if database_type not in DATABASE_PATHS:
                    return []

                db_path = DATABASE_PATHS[database_type]
                if not os.path.exists(db_path):
                    return []

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                similar_compounds = []

                # Получаем все соединения из базы данных
                cursor.execute(f"SELECT * FROM {database_type}")
                all_compounds = cursor.fetchall()

                # Получаем названия колонок
                cursor.execute(f"PRAGMA table_info({database_type})")
                columns = [row[1] for row in cursor.fetchall()]

                # Преобразуем в словари и очищаем данные
                compounds_data = []
                for row in all_compounds:
                    compound_dict = dict(zip(columns, row))
                    # Очищаем SMILES данные
                    cleaned_compound = self._clean_smiles_data(compound_dict)
                    compounds_data.append(cleaned_compound)

                conn.close()

            # Вычисляем схожесть для каждого соединения
            similarities = []
            for compound in compounds_data:
                # Пропускаем целевое соединение
                if compound.get('id') == target_compound.get('id'):
                    continue

                similarity_score = self._calculate_similarity(target_compound, compound, database_type)
                similarities.append((compound, similarity_score))

            # Сортируем по схожести и берем топ
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_compounds = [comp for comp, score in similarities[:limit] if score > 0]

            return similar_compounds

        except Exception as e:
            logger.error(f"Ошибка поиска похожих соединений: {e}")
            return []

    def _calculate_similarity(self, compound1: Dict[str, Any],
                            compound2: Dict[str, Any], database_type: str) -> float:
        """
        Расчет схожести между двумя соединениями
        """
        similarity = 0.0

        # 1. Схожесть по названию (текстовый поиск)
        name_sim = self._text_similarity(
            compound1.get('name', ''),
            compound2.get('name', '')
        )
        similarity += name_sim * 0.3

        # 2. Схожесть по формуле (для молекулярных соединений)
        if database_type in ['metabolites', 'carbohydrates', 'lipids']:
            formula_sim = self._formula_similarity(
                compound1.get('formula', ''),
                compound2.get('formula', '')
            )
            similarity += formula_sim * 0.4

        # 3. Схожесть по массе
        mass_sim = self._mass_similarity(
            compound1.get('exact_mass'),
            compound2.get('exact_mass')
        )
        similarity += mass_sim * 0.2

        # 4. Структурная схожесть (если доступен RDKit)
        if RDKIT_AVAILABLE and compound1.get('smiles') and compound2.get('smiles'):
            # Проверяем валидность SMILES перед расчетом
            if self._is_valid_smiles(compound1.get('smiles')) and self._is_valid_smiles(compound2.get('smiles')):
                struct_sim = self._structural_similarity(
                    compound1.get('smiles'),
                    compound2.get('smiles')
                )
                similarity += struct_sim * 0.5
            else:
                # Если SMILES невалидны, пропускаем структурное сравнение
                logger.debug(f"Пропускаю структурное сравнение: невалидные SMILES")

        # 5. Схожесть по свойствам
        if database_type == 'enzymes':
            prop_sim = self._enzyme_similarity(compound1, compound2)
            similarity += prop_sim * 0.3
        elif database_type == 'proteins':
            prop_sim = self._protein_similarity(compound1, compound2)
            similarity += prop_sim * 0.3

        return min(similarity, 1.0)  # Ограничиваем до 1.0

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Расчет текстового сходства"""
        if not text1 or not text2:
            return 0.0

        # Простой расчет на основе общих слов
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _formula_similarity(self, formula1: str, formula2: str) -> float:
        """Расчет сходства химических формул"""
        if not formula1 or not formula2:
            return 0.0

        # Простой парсинг элементов
        elements1 = self._parse_formula(formula1)
        elements2 = self._parse_formula(formula2)

        if not elements1 or not elements2:
            return 0.0

        # Jaccard similarity для множеств элементов
        set1 = set(elements1.keys())
        set2 = set(elements2.keys())

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        return len(intersection) / len(union) if union else 0.0

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """Парсинг химической формулы"""
        elements = {}
        # Простой парсер элементов
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula)

        for element, count in matches:
            count = int(count) if count else 1
            elements[element] = elements.get(element, 0) + count

        return elements

    def _mass_similarity(self, mass1: float, mass2: float) -> float:
        """Расчет сходства по молекулярной массе"""
        if mass1 is None or mass2 is None:
            return 0.0

        # Используем гауссово сходство
        diff = abs(mass1 - mass2)
        sigma = max(mass1, mass2) * 0.1  # 10% от большей массы

        return math.exp(-(diff ** 2) / (2 * sigma ** 2))

    def _structural_similarity(self, smiles1: str, smiles2: str) -> float:
        """Расчет структурного сходства с использованием RDKit"""
        if not RDKIT_AVAILABLE:
            return 0.0

        # Проверяем валидность SMILES
        if not self._is_valid_smiles(smiles1) or not self._is_valid_smiles(smiles2):
            return 0.0

        try:
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)

            if mol1 is None or mol2 is None:
                return 0.0

            # Morgan fingerprints
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=1024)

            # Tanimoto similarity
            return DataStructs.TanimotoSimilarity(fp1, fp2)

        except Exception as e:
            logger.error(f"Ошибка расчета структурного сходства: {e}")
            return 0.0

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Проверяет, является ли строка валидным SMILES"""
        if not smiles or not isinstance(smiles, str):
            return False
        
        # Проверяем на недопустимые значения
        invalid_values = ['0', 'None', 'null', '', 'nan', 'NaN']
        if smiles in invalid_values:
            logger.debug(f"Обнаружен невалидный SMILES: '{smiles}'")
            return False
        
        # Проверяем, что SMILES содержит хотя бы один символ
        if len(smiles.strip()) < 2:
            logger.debug(f"SMILES слишком короткий: '{smiles}'")
            return False
        
        # Проверяем, что SMILES содержит химические символы
        chemical_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}@+-=#$%:;,.')
        if not any(char in chemical_chars for char in smiles):
            logger.debug(f"SMILES не содержит химических символов: '{smiles}'")
            return False
        
        return True

    def _enzyme_similarity(self, enzyme1: Dict[str, Any], enzyme2: Dict[str, Any]) -> float:
        """Расчет сходства ферментов"""
        similarity = 0.0

        # Сходство по EC номеру
        ec1 = enzyme1.get('ec_number', '')
        ec2 = enzyme2.get('ec_number', '')

        if ec1 and ec2:
            # Сравнение первых цифр EC номера
            ec_parts1 = ec1.split('.')
            ec_parts2 = ec2.split('.')

            matches = 0
            for i in range(min(len(ec_parts1), len(ec_parts2))):
                if ec_parts1[i] == ec_parts2[i]:
                    matches += 1
                else:
                    break

            similarity += (matches / 4.0) * 0.5  # EC номер имеет 4 уровня

        # Сходство по семейству
        family1 = enzyme1.get('family', '')
        family2 = enzyme2.get('family', '')

        if family1 and family2:
            if family1.lower() == family2.lower():
                similarity += 0.3

        return similarity

    def _protein_similarity(self, protein1: Dict[str, Any], protein2: Dict[str, Any]) -> float:
        """Расчет сходства белков"""
        similarity = 0.0

        # Сходство по функции
        func1 = protein1.get('function', '')
        func2 = protein2.get('function', '')

        if func1 and func2:
            func_sim = self._text_similarity(func1, func2)
            similarity += func_sim * 0.4

        # Сходство по семейству
        family1 = protein1.get('family', '')
        family2 = protein2.get('family', '')

        if family1 and family2:
            if family1.lower() == family2.lower():
                similarity += 0.3

        # Сходство по организму
        org1 = protein1.get('organism', '')
        org2 = protein2.get('organism', '')

        if org1 and org2:
            if org1.lower() == org2.lower():
                similarity += 0.3

        return similarity

    def _apply_filters(self, compounds: List[Dict[str, Any]], mass_range: tuple,
                      smiles_only: bool, keyword_filter: str,
                      formula_elements: List[str]) -> List[Dict[str, Any]]:
        """Применяет фильтры к списку соединений"""
        filtered_compounds = []

        for compound in compounds:
            # Фильтр по массе
            mass = compound.get('exact_mass')
            if mass is not None and (mass < mass_range[0] or mass > mass_range[1]):
                continue

            # Фильтр по SMILES
            if smiles_only and not self._is_valid_smiles(compound.get('smiles', '')):
                continue

            # Фильтр по ключевым словам
            if keyword_filter:
                keywords = [kw.strip().lower() for kw in keyword_filter.split(',')]
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

        return filtered_compounds

    def _create_clustering_features(self, compounds: List[Dict[str, Any]],
                                   database_type: str) -> Tuple[List[List[float]], List[str]]:
        """Создание расширенных признаков для кластеризации"""
        features = []
        feature_names = []

        # Определяем базовые признаки
        base_features = ['mass', 'name_length', 'formula_length']
        feature_names.extend(base_features)

        for compound in compounds:
            feature_vector = []

            # Базовые признаки
            mass = compound.get('exact_mass', 0)
            feature_vector.append(float(mass) if mass else 0.0)

            name_len = len(compound.get('name', ''))
            feature_vector.append(float(name_len))

            formula_len = len(compound.get('formula', ''))
            feature_vector.append(float(formula_len))

            # Специфические признаки по типам
            if database_type in ['metabolites', 'carbohydrates', 'lipids']:
                # Химические признаки
                elements = self._parse_formula(compound.get('formula', ''))
                feature_vector.append(float(len(elements)))  # Количество элементов
                feature_vector.append(float(sum(elements.values())))  # Общее количество атомов

                # Структурные признаки
                class_len = len(compound.get('class_name', ''))
                feature_vector.append(float(class_len))

                feature_names.extend(['elements_count', 'atoms_total', 'class_length'])

                # Молекулярные дескрипторы (если есть SMILES)
                if RDKIT_AVAILABLE and compound.get('smiles'):
                    try:
                        mol = Chem.MolFromSmiles(compound.get('smiles'))
                        if mol:
                            feature_vector.append(float(Chem.rdMolDescriptors.CalcNumHBD(mol)))  # H-bond donors
                            feature_vector.append(float(Chem.rdMolDescriptors.CalcNumHBA(mol)))  # H-bond acceptors
                            feature_vector.append(float(Descriptors.MolLogP(mol)))  # LogP
                            feature_vector.append(float(Descriptors.MolWt(mol)))   # Molecular weight
                            feature_names.extend(['hbd', 'hba', 'logp', 'molwt'])
                        else:
                            # Добавляем нули если SMILES невалиден
                            feature_vector.extend([0.0, 0.0, 0.0, 0.0])
                            feature_names.extend(['hbd', 'hba', 'logp', 'molwt'])
                    except:
                        feature_vector.extend([0.0, 0.0, 0.0, 0.0])
                        feature_names.extend(['hbd', 'hba', 'logp', 'molwt'])
                else:
                    feature_vector.extend([0.0, 0.0, 0.0, 0.0])
                    feature_names.extend(['hbd', 'hba', 'logp', 'molwt'])

            elif database_type == 'enzymes':
                # EC номер признаки
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

                feature_names.extend(['ec1', 'ec2', 'ec3', 'ec4'])

                # Функциональные признаки
                family_len = len(compound.get('family', ''))
                feature_vector.append(float(family_len))
                feature_names.append('family_length')

            elif database_type == 'proteins':
                # Белковые признаки
                seq_len = compound.get('sequence_length', 0)
                feature_vector.append(float(seq_len) if seq_len else 0.0)

                func_len = len(compound.get('function', ''))
                feature_vector.append(float(func_len))

                family_len = len(compound.get('family', ''))
                feature_vector.append(float(family_len))

                feature_names.extend(['seq_length', 'func_length', 'family_length'])

            features.append(feature_vector)

        # Убираем дубликаты из названий признаков
        feature_names = list(dict.fromkeys(feature_names))

        return features, feature_names

    def save_recommendation_session(self, session_data: Dict[str, Any], session_name: str) -> bool:
        """Сохранение сессии рекомендаций"""
        try:
            import json
            import os
            from datetime import datetime

            # Создаем директорию для сессий если её нет
            sessions_dir = os.path.join(os.path.dirname(__file__), '..', 'sessions')
            os.makedirs(sessions_dir, exist_ok=True)

            # Добавляем метаданные сессии
            session_data['metadata'] = {
                'name': session_name,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }

            # Сохраняем сессию
            session_file = os.path.join(sessions_dir, f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2, default=str)

            logger.info(f"Сессия рекомендаций сохранена: {session_file}")
            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения сессии: {e}")
            return False

    def load_recommendation_sessions(self) -> List[Dict[str, Any]]:
        """Загрузка списка сохраненных сессий"""
        try:
            import os
            import json
            from datetime import datetime

            sessions_dir = os.path.join(os.path.dirname(__file__), '..', 'sessions')
            if not os.path.exists(sessions_dir):
                return []

            sessions = []
            for filename in os.listdir(sessions_dir):
                if filename.endswith('.json'):
                    try:
                        filepath = os.path.join(sessions_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)

                        # Добавляем информацию о файле
                        session_data['file_info'] = {
                            'filename': filename,
                            'filepath': filepath,
                            'file_size': os.path.getsize(filepath)
                        }

                        sessions.append(session_data)
                    except Exception as e:
                        logger.warning(f"Ошибка загрузки сессии {filename}: {e}")

            # Сортируем по времени создания (новые сверху)
            sessions.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
            return sessions

        except Exception as e:
            logger.error(f"Ошибка загрузки сессий: {e}")
            return []

    def delete_recommendation_session(self, session_filename: str) -> bool:
        """Удаление сессии рекомендаций"""
        try:
            import os
            sessions_dir = os.path.join(os.path.dirname(__file__), '..', 'sessions')
            filepath = os.path.join(sessions_dir, session_filename)

            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Сессия удалена: {session_filename}")
                return True
            else:
                logger.warning(f"Файл сессии не найден: {session_filename}")
                return False

        except Exception as e:
            logger.error(f"Ошибка удаления сессии: {e}")
            return False

    def cluster_compounds(self, compounds: List[Dict[str, Any]],
                         database_type: str, n_clusters: int = 5,
                         algorithm: str = "kmeans") -> Dict[str, Any]:
        """Улучшенная кластеризация соединений с множеством алгоритмов"""
        if not compounds or len(compounds) < 2:
            return {"error": "Недостаточно данных для кластеризации"}

        if len(compounds) < n_clusters and algorithm == "kmeans":
            n_clusters = max(2, len(compounds) // 2)  # Автоматическая корректировка

        try:
            # Создаем расширенные признаки для кластеризации
            features, feature_names = self._create_clustering_features(compounds, database_type)

            if not features:
                return {"error": "Недостаточно данных для создания признаков"}

            # Нормализация признаков
            import numpy as np
            features_array = np.array(features)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Выполняем кластеризацию выбранным алгоритмом
            if algorithm == "kmeans":
                from sklearn.cluster import KMeans
                if len(compounds) < n_clusters:
                    n_clusters = max(2, len(compounds) // 2)
                cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = cluster_model.fit_predict(features_scaled)
                centroids = cluster_model.cluster_centers_

            elif algorithm == "dbscan":
                from sklearn.cluster import DBSCAN
                # Автоматический подбор eps на основе данных
                from sklearn.neighbors import NearestNeighbors
                neighbors = NearestNeighbors(n_neighbors=5)
                neighbors_fit = neighbors.fit(features_scaled)
                distances, indices = neighbors_fit.kneighbors(features_scaled)
                distances = np.sort(distances, axis=0)
                distances = distances[:, 1]
                eps = np.percentile(distances, 90)  # 90-й процентиль
                cluster_model = DBSCAN(eps=eps, min_samples=max(2, len(compounds) // 20))
                clusters = cluster_model.fit_predict(features_scaled)
                centroids = None
                # Пересчитываем количество кластеров (DBSCAN может дать -1 для шумовых точек)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

            elif algorithm == "agglomerative":
                from sklearn.cluster import AgglomerativeClustering
                cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
                clusters = cluster_model.fit_predict(features_scaled)
                centroids = None

            else:
                return {"error": f"Неподдерживаемый алгоритм кластеризации: {algorithm}"}

            # Вычисляем метрики качества кластеризации
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            quality_metrics = {}
            try:
                if n_clusters > 1 and len(set(clusters)) > 1:
                    quality_metrics['silhouette'] = silhouette_score(features_scaled, clusters)
                    quality_metrics['calinski_harabasz'] = calinski_harabasz_score(features_scaled, clusters)
                    quality_metrics['davies_bouldin'] = davies_bouldin_score(features_scaled, clusters)
                else:
                    quality_metrics = {"warning": "Недостаточно кластеров для оценки качества"}
            except:
                quality_metrics = {"warning": "Не удалось вычислить метрики качества"}

            # Группировка результатов
            cluster_results = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_results:
                    cluster_results[cluster_id] = []
                cluster_results[cluster_id].append(compounds[i])

            return {
                "clusters": cluster_results,
                "n_clusters": n_clusters,
                "total_compounds": len(compounds),
                "algorithm": algorithm,
                "quality_metrics": quality_metrics,
                "feature_names": feature_names,
                "centroids": centroids.tolist() if centroids is not None else None
            }

        except Exception as e:
            logger.error(f"Ошибка кластеризации: {e}")
            return {"error": str(e)}

    def get_recommendation_explanation(self, target_compound: Dict[str, Any],
                                     similar_compound: Dict[str, Any],
                                     similarity_score: float) -> str:
        """Генерация объяснения рекомендации"""
        explanations = []

        # Объяснение по названию
        name1 = target_compound.get('name', '').lower()
        name2 = similar_compound.get('name', '').lower()

        if name1 and name2:
            common_words = set(name1.split()) & set(name2.split())
            if common_words:
                explanations.append(f"Общие слова в названиях: {', '.join(common_words)}")

        # Объяснение по формуле
        formula1 = target_compound.get('formula', '')
        formula2 = similar_compound.get('formula', '')

        if formula1 and formula2:
            elements1 = set(self._parse_formula(formula1).keys())
            elements2 = set(self._parse_formula(formula2).keys())
            common_elements = elements1 & elements2

            if common_elements:
                explanations.append(f"Общие химические элементы: {', '.join(common_elements)}")

        # Объяснение по массе
        mass1 = target_compound.get('exact_mass')
        mass2 = similar_compound.get('exact_mass')

        if mass1 and mass2:
            mass_diff = abs(mass1 - mass2)
            if mass_diff < 10:
                explanations.append(f"Очень близкие массы: {mass1:.1f} vs {mass2:.1f} Da")
            elif mass_diff < 100:
                explanations.append(f"Близкие массы: {mass1:.1f} vs {mass2:.1f} Da")
        # Объяснение по структуре
        if RDKIT_AVAILABLE and similarity_score > 0.7:
            explanations.append("Высокая структурная схожесть молекул")

        if not explanations:
            explanations.append(f"Общий коэффициент схожести: {similarity_score:.2f}")

        return "; ".join(explanations)


def render_recommendations_interface():
    """Отобразить интерфейс системы рекомендаций"""
    st.header("🎯 Система рекомендаций")

    engine = RecommendationsEngine()

    # Загрузка сохраненных сессий
    with st.expander("📂 Загрузка сохраненных сессий"):
        saved_sessions = engine.load_recommendation_sessions()

        if saved_sessions:
            session_options = [f"{s.get('metadata', {}).get('name', 'Без названия')} - {s.get('metadata', {}).get('timestamp', '')[:19]}" for s in saved_sessions]
            selected_session_idx = st.selectbox(
                "Выберите сессию для загрузки:",
                options=range(len(saved_sessions)),
                format_func=lambda x: session_options[x] if x < len(session_options) else "—"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 Загрузить сессию", use_container_width=True):
                    session_data = saved_sessions[selected_session_idx]
                    st.session_state.recommendation_results = session_data.get('recommendations', [])
                    st.session_state.target_compound = session_data.get('target_compound', {})
                    st.session_state.filters_applied = session_data.get('filters', {})
                    st.success("✅ Сессия загружена! Просмотрите результаты ниже.")

            with col2:
                if st.button("🗑️ Удалить сессию", use_container_width=True):
                    filename = saved_sessions[selected_session_idx].get('file_info', {}).get('filename')
                    if filename and engine.delete_recommendation_session(filename):
                        st.success("✅ Сессия удалена!")
                        st.rerun()  # Обновляем интерфейс
                    else:
                        st.error("❌ Ошибка удаления сессии")
        else:
            st.info("💡 У вас пока нет сохраненных сессий. Выполните поиск рекомендаций и сохраните результаты.")

    st.divider()

    # Выбор типа базы данных для поиска
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
        # Загружаем соединения из выбранной базы
        try:
            # Try absolute import first
            from config.settings import DATABASE_PATHS
        except ImportError:
            # Fallback to relative import
            from ..config.settings import DATABASE_PATHS

            db_path = DATABASE_PATHS[selected_db]
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    # Проверяем структуру таблицы
                    cursor.execute(f"PRAGMA table_info({selected_db})")
                    columns = [row[1] for row in cursor.fetchall()]

                    if not columns:
                        st.error(f"❌ Ошибка: таблица {selected_db} пустая или не существует")
                        return

                    # Получаем все соединения (увеличен лимит для лучших рекомендаций)
                    cursor.execute(f"SELECT * FROM {selected_db} LIMIT 1000")  # Увеличено для лучших рекомендаций
                    compounds = cursor.fetchall()

                    compounds_list = [dict(zip(columns, row)) for row in compounds]
                    conn.close()

                    if not compounds_list:
                        st.warning(f"⚠️ В таблице {selected_db} нет данных")
                        return

                except sqlite3.Error as e:
                    st.error(f"❌ Ошибка базы данных: {str(e)}")
                    logger.error(f"Database error for {selected_db}: {e}")
                    return
                except Exception as e:
                    st.error(f"❌ Ошибка загрузки данных: {str(e)}")
                    logger.error(f"Error loading data from {selected_db}: {e}")
                    return
            else:
                st.error(f"❌ Файл базы данных {db_path} не найден")
                return

            if compounds_list:
                    st.success(f"✅ Загружено {len(compounds_list)} соединений из базы {database_options[selected_db]}")

                    # Очищаем SMILES данные для всех соединений
                    cleaned_compounds = []
                    for compound in compounds_list:
                        cleaned_compound = engine._clean_smiles_data(compound)
                        cleaned_compounds.append(cleaned_compound)
                    
                    compounds_list = cleaned_compounds

                    # Выбор целевого соединения
                    compound_names = [f"{c.get('name', 'Без названия')} (ID: {c.get('id', '—')})" for c in compounds_list]
                    selected_compound_idx = st.selectbox(
                        "Выберите соединение для поиска аналогов:",
                        options=range(len(compounds_list)),
                        format_func=lambda x: compound_names[x]
                    )

                    target_compound = compounds_list[selected_compound_idx]

                    # Параметры поиска
                    st.subheader("⚙️ Параметры поиска")

                    # Основные параметры
                    col1, col2 = st.columns(2)
                    with col1:
                        limit = st.slider("Количество рекомендаций:", 5, 50, 10)
                    with col2:
                        min_similarity = st.slider("Минимальная схожесть (%):", 0, 100, 30) / 100.0

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

                        with col4:
                            # Фильтр по наличию SMILES
                            smiles_only = st.checkbox(
                                "Только с SMILES",
                                help="Показывать только соединения с валидными SMILES для структурного анализа"
                            )

                        # Фильтр по ключевым словам
                        keyword_filter = st.text_input(
                            "Ключевые слова в названии:",
                            placeholder="например: glucose, dehydrogenase",
                            help="Фильтровать по словам в названии (через запятую)"
                        )

                        # Фильтр по элементам в формуле
                        formula_elements = st.multiselect(
                            "Обязательные элементы в формуле:",
                            options=["C", "H", "O", "N", "P", "S", "Cl", "Br", "I", "F"],
                            help="Соединения должны содержать выбранные химические элементы"
                        )

                    # Поиск рекомендаций
                    if st.button("🔍 Найти похожие соединения", type="primary", width='stretch'):
                        with st.spinner("Ищу похожие соединения..."):
                            # Применяем фильтры к списку соединений перед поиском
                            filtered_compounds_list = engine._apply_filters(
                                compounds_list, mass_range, smiles_only, keyword_filter, formula_elements
                            )

                            if len(filtered_compounds_list) < 2:
                                st.warning("⚠️ После применения фильтров осталось слишком мало соединений для поиска рекомендаций")
                                return

                            similar_compounds = engine.find_similar_compounds(
                                target_compound, selected_db, limit, filtered_compounds_list
                            )

                            # Фильтруем по минимальной схожести
                            final_filtered_compounds = []
                            for comp in similar_compounds:
                                similarity = engine._calculate_similarity(target_compound, comp, selected_db)
                                if similarity >= min_similarity:
                                    final_filtered_compounds.append((comp, similarity))

                            st.session_state.recommendation_results = final_filtered_compounds
                            st.session_state.target_compound = target_compound
                            st.session_state.filters_applied = {
                                'mass_range': mass_range,
                                'smiles_only': smiles_only,
                                'keyword_filter': keyword_filter,
                                'formula_elements': formula_elements
                            }

                            # Сохранение сессии
                            st.divider()
                            st.subheader("💾 Сохранение сессии")

                            session_name = st.text_input(
                                "Название сессии:",
                                placeholder="Моя сессия рекомендаций",
                                help="Введите название для сохранения текущей сессии"
                            )

                            if st.button("💾 Сохранить сессию", type="secondary", use_container_width=True):
                                if session_name.strip():
                                    session_data = {
                                        'recommendations': st.session_state.recommendation_results,
                                        'target_compound': st.session_state.target_compound,
                                        'filters': st.session_state.filters_applied,
                                        'database_type': selected_db,
                                        'timestamp': pd.Timestamp.now().isoformat()
                                    }

                                    if engine.save_recommendation_session(session_data, session_name.strip()):
                                        st.success(f"✅ Сессия '{session_name}' успешно сохранена!")
                                    else:
                                        st.error("❌ Ошибка сохранения сессии")
                                else:
                                    st.warning("⚠️ Введите название сессии")

                    # Отображение результатов
                    if 'recommendation_results' in st.session_state and st.session_state.recommendation_results:
                        results = st.session_state.recommendation_results
                        target = st.session_state.target_compound

                        st.subheader(f"🎯 Рекомендации для: {target.get('name', 'Без названия')}")

                        # Таблица результатов
                        result_data = []
                        for comp, similarity in results:
                            result_data.append({
                                "Название": comp.get('name', '—'),
                                "Схожесть": f"{similarity:.1%}",
                                "Формула": comp.get('formula', '—'),
                                "Масса (Da)": f"{comp.get('exact_mass', 0):.2f}" if comp.get('exact_mass') else "—",
                                "Объяснение": engine.get_recommendation_explanation(target, comp, similarity)
                            })

                        if result_data:
                            df = pd.DataFrame(result_data)

                            st.dataframe(
                                df,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    "Название": st.column_config.TextColumn(width="large"),
                                    "Схожесть": st.column_config.TextColumn(width="small"),
                                    "Формула": st.column_config.TextColumn(width="medium"),
                                    "Масса (Da)": st.column_config.NumberColumn(width="small", format="%.2f"),
                                    "Объяснение": st.column_config.TextColumn(width="large")
                                }
                            )

                            # Кнопки действий для каждого результата
                            st.subheader("🔧 Действия с результатами")
                            
                            for i, (comp, similarity) in enumerate(results):
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    
                                    with col1:
                                        st.markdown(f"**{comp.get('name', '—')}** (схожесть: {similarity:.1%})")
                                    
                                    with col2:
                                        # Кнопка показать детали
                                        if st.button(f"📋 Детали", key=f"details_{i}_{comp.get('id', i)}", use_container_width=True):
                                            # Открываем диалог деталей
                                            try:
                                                from ..main import open_dialog_safely
                                            except ImportError:
                                                try:
                                                    from main import open_dialog_safely
                                                except ImportError:
                                                    # Fallback функция
                                                    def open_dialog_safely(dialog_type: str, entity: Dict[str, Any]):
                                                        st.session_state[f"show_{dialog_type}_details"] = True
                                                        st.session_state[f"selected_{dialog_type}"] = entity
                                                    
                                            # Определяем тип соединения
                                            if selected_db == "metabolites":
                                                open_dialog_safely("metabolite", comp)
                                            elif selected_db == "enzymes":
                                                open_dialog_safely("enzyme", comp)
                                            elif selected_db == "proteins":
                                                open_dialog_safely("protein", comp)
                                            elif selected_db == "carbohydrates":
                                                open_dialog_safely("carbohydrate", comp)
                                            elif selected_db == "lipids":
                                                open_dialog_safely("lipid", comp)
                                        
                                        # Кнопка добавить к сравнению
                                        if st.button(f"⚖️ Сравнить", key=f"compare_{i}_{comp.get('id', i)}", use_container_width=True):
                                            try:
                                                from .comparison import add_to_comparison_button, comparison_comparator
                                                # Добавляем к сравнению
                                                add_to_comparison_button(comp, selected_db, comparison_comparator)
                                                st.success(f"✅ {comp.get('name', 'Соединение')} добавлено к сравнению")
                                            except Exception as e:
                                                st.error(f"Ошибка добавления к сравнению: {e}")
                                    
                                    st.divider()

                            # Диаграмма схожести
                            st.subheader("📊 Визуализация схожести")

                            similarities = [sim for _, sim in results]
                            names = [comp.get('name', '—')[:30] for comp, _ in results]  # Ограничиваем длину

                            if similarities:
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=names,
                                        y=[s * 100 for s in similarities],
                                        marker_color='#1f77b4',
                                        text=[f'{s*100:.1f}%' for s in similarities],
                                        textposition='auto'
                                    )
                                ])

                                fig.update_layout(
                                    title="Схожесть найденных соединений",
                                    xaxis_title="Соединение",
                                    yaxis_title="Схожесть (%)",
                                    height=400,
                                    xaxis_tickangle=-45
                                )

                                st.plotly_chart(fig, width='stretch')
                                
                                # Кнопка экспорта результатов
                                st.subheader("💾 Экспорт результатов")
                                csv_data = df.to_csv(index=False, encoding='utf-8')
                                st.download_button(
                                    label="📥 Скачать результаты (CSV)",
                                    data=csv_data,
                                    file_name=f"recommendations_{selected_db}_{target.get('name', 'compound')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )

                    # Кластеризация
                    st.divider()
                    st.subheader("📈 Улучшенная кластеризация соединений")

                    col1, col2 = st.columns(2)
                    with col1:
                        n_clusters = st.slider("Количество кластеров:", 2, 10, 5)
                    with col2:
                        algorithm_options = {
                            "kmeans": "K-means (сферические кластеры)",
                            "dbscan": "DBSCAN (произвольная форма)",
                            "agglomerative": "Иерархическая (древовидная)"
                        }
                        selected_algorithm = st.selectbox(
                            "Алгоритм кластеризации:",
                            options=list(algorithm_options.keys()),
                            format_func=lambda x: algorithm_options[x]
                        )

                    if st.button("🎯 Выполнить кластеризацию", width='stretch'):
                        with st.spinner("Выполняю кластеризацию..."):
                            cluster_results = engine.cluster_compounds(
                                compounds_list, selected_db, n_clusters, selected_algorithm
                            )

                            if "error" not in cluster_results:
                                st.success(f"✅ Найдено {cluster_results['n_clusters']} кластеров из {cluster_results['total_compounds']} соединений")
                                st.info(f"🎯 Алгоритм: {algorithm_options[selected_algorithm]}")

                                # Отображение метрик качества
                                if "quality_metrics" in cluster_results and "warning" not in cluster_results["quality_metrics"]:
                                    metrics = cluster_results["quality_metrics"]
                                    st.subheader("📊 Метрики качества кластеризации")

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if 'silhouette' in metrics:
                                            st.metric(
                                                "Коэффициент силуэта",
                                                f"{metrics['silhouette']:.3f}",
                                                help="Чем ближе к 1, тем лучше разделение кластеров"
                                            )
                                    with col2:
                                        if 'calinski_harabasz' in metrics:
                                            st.metric(
                                                "Индекс Calinski-Harabasz",
                                                f"{metrics['calinski_harabasz']:.1f}",
                                                help="Чем выше значение, тем лучше кластеризация"
                                            )
                                    with col3:
                                        if 'davies_bouldin' in metrics:
                                            st.metric(
                                                "Индекс Davies-Bouldin",
                                                f"{metrics['davies_bouldin']:.3f}",
                                                help="Чем ниже значение, тем лучше кластеризация"
                                            )
                                elif "quality_metrics" in cluster_results and "warning" in cluster_results["quality_metrics"]:
                                    st.warning(cluster_results["quality_metrics"]["warning"])

                                # Отображение кластеров
                                clusters = cluster_results['clusters']
                                
                                # Визуализация распределения кластеров
                                st.subheader("📊 Распределение кластеров")

                                cluster_sizes = [len(cluster_compounds) for cluster_compounds in clusters.values()]
                                cluster_labels = [f"Кластер {cluster_id + 1}" for cluster_id in clusters.keys()]

                                # Круговая диаграмма
                                fig_pie = go.Figure(data=[
                                    go.Pie(
                                        labels=cluster_labels,
                                        values=cluster_sizes,
                                        textinfo='label+percent',
                                        insidetextorientation='radial'
                                    )
                                ])

                                fig_pie.update_layout(
                                    title="Распределение соединений по кластерам",
                                    height=400
                                )

                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.plotly_chart(fig_pie, width='stretch')

                                with col2:
                                    # Статистика кластеров
                                    st.subheader("📈 Статистика")
                                    st.metric("Всего кластеров", len(clusters))
                                    st.metric("Максимальный кластер", max(cluster_sizes))
                                    st.metric("Минимальный кластер", min(cluster_sizes))
                                    avg_size = sum(cluster_sizes) / len(cluster_sizes)
                                    st.metric("Средний размер", ".1f")

                                # Подготовка данных для экспорта
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

                                # Кнопка экспорта кластеров
                                if export_data:
                                    st.subheader("💾 Экспорт кластеров")
                                    export_df = pd.DataFrame(export_data)
                                    csv_cluster_data = export_df.to_csv(index=False, encoding='utf-8')
                                    st.download_button(
                                        label="📥 Скачать кластеры (CSV)",
                                        data=csv_cluster_data,
                                        file_name=f"clusters_{selected_db}_{n_clusters}_clusters.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )

                                for cluster_id, cluster_compounds in clusters.items():
                                    with st.expander(f"Кластер {cluster_id + 1} ({len(cluster_compounds)} соединений)"):
                                        cluster_data = []
                                        for comp in cluster_compounds[:10]:  # Показываем первые 10
                                            cluster_data.append({
                                                "Название": comp.get('name', '—'),
                                                "Формула": comp.get('formula', '—'),
                                                "Масса": f"{comp.get('exact_mass', 0):.2f}" if comp.get('exact_mass') else "—"
                                            })

                                        if cluster_data:
                                            cluster_df = pd.DataFrame(cluster_data)
                                            st.dataframe(cluster_df, width='stretch', hide_index=True)
                                            
                                            # Кнопки действий для соединений в кластере
                                            st.subheader("🔧 Действия с соединениями кластера")
                                            
                                            for i, comp in enumerate(cluster_compounds[:10]):
                                                with st.container():
                                                    col1, col2 = st.columns([3, 1])
                                                    
                                                    with col1:
                                                        st.markdown(f"**{comp.get('name', '—')}**")
                                                    
                                                    with col2:
                                                        # Кнопка показать детали
                                                        if st.button(f"📋 Детали", key=f"cluster_details_{cluster_id}_{i}_{comp.get('id', i)}", use_container_width=True):
                                                            # Открываем диалог деталей
                                                            try:
                                                                from ..main import open_dialog_safely
                                                            except ImportError:
                                                                try:
                                                                    from main import open_dialog_safely
                                                                except ImportError:
                                                                    # Fallback функция
                                                                    def open_dialog_safely(dialog_type: str, entity: Dict[str, Any]):
                                                                        st.session_state[f"show_{dialog_type}_details"] = True
                                                                        st.session_state[f"selected_{dialog_type}"] = entity
                                                                    
                                                            # Определяем тип соединения
                                                            if selected_db == "metabolites":
                                                                open_dialog_safely("metabolite", comp)
                                                            elif selected_db == "enzymes":
                                                                open_dialog_safely("enzyme", comp)
                                                            elif selected_db == "proteins":
                                                                open_dialog_safely("protein", comp)
                                                            elif selected_db == "carbohydrates":
                                                                open_dialog_safely("carbohydrate", comp)
                                                            elif selected_db == "lipids":
                                                                open_dialog_safely("lipid", comp)
                                                            
                                                        # Кнопка добавить к сравнению
                                                        if st.button(f"⚖️ Сравнить", key=f"cluster_compare_{cluster_id}_{i}_{comp.get('id', i)}", use_container_width=True):
                                                            try:
                                                                from .comparison import add_to_comparison_button, comparison_comparator
                                                                # Добавляем к сравнению
                                                                add_to_comparison_button(comp, selected_db, comparison_comparator)
                                                                st.success(f"✅ {comp.get('name', 'Соединение')} добавлено к сравнению")
                                                            except Exception as e:
                                                                st.error(f"Ошибка добавления к сравнению: {e}")
                                                    
                                                    st.divider()

                            else:
                                st.error(f"❌ Ошибка кластеризации: {cluster_results['error']}")

            else:
                st.warning("В выбранной базе данных нет соединений")

        except Exception as e:
            st.error(f"Ошибка загрузки данных: {str(e)}")

    # Информация о системе рекомендаций
    with st.expander("ℹ️ О системе рекомендаций"):
        st.markdown("""
        **🎯 Система рекомендаций** находит похожие соединения на основе:

        - **Текстового сходства**: общие слова в названиях
        - **Химического состава**: общие элементы в формулах
        - **Молекулярной массы**: близкие значения масс
        - **Структурного сходства**: молекулярные fingerprints (требует RDKit)
        - **Функциональных свойств**: для ферментов и белков

        **Кластеризация** группирует соединения по схожим характеристикам.

        **Алгоритмы**: TfidfVectorizer, K-means, Tanimoto similarity, Morgan fingerprints.
        """)
