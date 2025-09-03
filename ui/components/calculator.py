"""
Калькулятор молекулярных свойств
Расчет физико-химических свойств молекул в реальном времени
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

# Проверяем доступность RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    from rdkit.Chem.MolSurf import TPSA
    RDKIT_AVAILABLE = True
    logger.info("RDKit доступен для калькулятора")
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit не доступен - базовые функции калькулятора будут ограничены")


class MolecularCalculator:
    """Класс для расчета молекулярных свойств"""

    def __init__(self):
        self.last_smiles = None
        self.last_properties = {}

    def parse_smiles(self, smiles: str) -> Tuple[bool, Optional[object], str]:
        """Парсинг SMILES строки"""
        if not smiles or not smiles.strip():
            return False, None, "Введите SMILES строку"

        smiles = smiles.strip()

        if not RDKIT_AVAILABLE:
            return False, None, "RDKit не установлен. Установите rdkit для расширенных расчетов."

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, None, "Неверный формат SMILES"
            return True, mol, ""
        except Exception as e:
            return False, None, f"Ошибка парсинга SMILES: {str(e)}"

    def calculate_basic_properties(self, smiles: str) -> Dict[str, Any]:
        """Расчет основных свойств молекулы"""
        success, mol, error_msg = self.parse_smiles(smiles)

        if not success:
            return {"error": error_msg}

        if not RDKIT_AVAILABLE:
            # Базовые расчеты без RDKit
            return {
                "formula": self._extract_formula_from_smiles(smiles),
                "smiles": smiles,
                "note": "Расширенные расчеты требуют установки RDKit"
            }

        try:
            properties = {
                "smiles": smiles,
                "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
                "molecular_weight": round(Descriptors.MolWt(mol), 3),
                "exact_mass": round(Descriptors.ExactMolWt(mol), 6),
                "heavy_atom_count": Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol),
                "atom_count": mol.GetNumAtoms(),
                "bond_count": mol.GetNumBonds(),
                "ring_count": Chem.rdMolDescriptors.CalcNumRings(mol),
                "aromatic_rings": Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
                "rotatable_bonds": Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
                "h_bond_donors": Chem.rdMolDescriptors.CalcNumHBD(mol),
                "h_bond_acceptors": Chem.rdMolDescriptors.CalcNumHBA(mol),
                "tpsa": round(TPSA(mol), 2),  # Topological Polar Surface Area
                "logp": round(Crippen.MolLogP(mol), 2),
                "clogp": round(Crippen.MolLogP(mol), 2),  # cLogP (same as LogP in RDKit)
                "solubility": self._estimate_solubility(Crippen.MolLogP(mol), TPSA(mol)),
            }

            # Расчет изотопных паттернов (упрощенный)
            properties["isotopic_pattern"] = self._calculate_isotopic_pattern(mol)

            # Оценка токсичности (простая эвристика)
            properties["toxicity_estimate"] = self._estimate_toxicity(mol)

            return properties

        except Exception as e:
            logger.error(f"Ошибка расчета свойств: {e}")
            return {"error": f"Ошибка расчета: {str(e)}"}

    def _extract_formula_from_smiles(self, smiles: str) -> str:
        """Простая экстракция формулы из SMILES (без RDKit)"""
        # Упрощенная логика для базовой формулы
        formula_dict = {}

        # Простой парсинг элементов
        elements = re.findall(r'([A-Z][a-z]?)', smiles)
        for element in elements:
            formula_dict[element] = formula_dict.get(element, 0) + 1

        # Формируем формулу
        formula_parts = []
        for element in sorted(formula_dict.keys()):
            count = formula_dict[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")

        return "".join(formula_parts)

    def _estimate_solubility(self, logp: float, tpsa: float) -> str:
        """Оценка растворимости на основе LogP и TPSA"""
        if logp < 0 and tpsa > 100:
            return "Высокая растворимость в воде"
        elif logp > 3:
            return "Низкая растворимость в воде"
        elif logp > 0:
            return "Средняя растворимость в воде"
        else:
            return "Хорошая растворимость в воде"

    def _calculate_isotopic_pattern(self, mol) -> Dict[str, float]:
        """Расчет основных изотопных паттернов"""
        try:
            # Получаем атомный состав
            atom_counts = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

            # Основные изотопы (упрощенная модель)
            isotopic_data = {}

            if 'C' in atom_counts:
                isotopic_data['¹²C'] = atom_counts['C']
                isotopic_data['¹³C'] = atom_counts['C'] * 0.011  # Природная распространенность

            if 'H' in atom_counts:
                isotopic_data['¹H'] = atom_counts['H']
                isotopic_data['²H'] = atom_counts['H'] * 0.00015

            if 'O' in atom_counts:
                isotopic_data['¹⁶O'] = atom_counts['O']
                isotopic_data['¹⁸O'] = atom_counts['O'] * 0.002

            return isotopic_data

        except Exception as e:
            return {"error": f"Ошибка расчета изотопов: {str(e)}"}

    def _estimate_toxicity(self, mol) -> str:
        """Простая оценка токсичности"""
        try:
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)

            # Простые правила Липинского
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1

            if violations == 0:
                return "Низкая токсичность (правило 5)"
            elif violations == 1:
                return "Средняя токсичность"
            else:
                return "Высокая токсичность"

        except:
            return "Не определена"

    def create_properties_table(self, properties: Dict[str, Any]) -> pd.DataFrame:
        """Создание таблицы свойств"""
        if "error" in properties:
            return pd.DataFrame({"Свойство": ["Ошибка"], "Значение": [properties["error"]]})

        # Группировка свойств
        basic_props = {
            "Формула": properties.get("formula", "—"),
            "Молекулярная масса": f"{properties.get('molecular_weight', '—')} Da",
            "Точная масса": f"{properties.get('exact_mass', '—')} Da",
            "Количество атомов": properties.get("atom_count", "—"),
            "Количество тяжелых атомов": properties.get("heavy_atom_count", "—"),
            "Количество связей": properties.get("bond_count", "—"),
            "Количество колец": properties.get("ring_count", "—"),
            "Ароматических колец": properties.get("aromatic_rings", "—"),
        }

        physchem_props = {
            "LogP": properties.get("logp", "—"),
            "TPSA": f"{properties.get('tpsa', '—')} Å²",
            "Вращаемых связей": properties.get("rotatable_bonds", "—"),
            "Доноров водородных связей": properties.get("h_bond_donors", "—"),
            "Акцепторов водородных связей": properties.get("h_bond_acceptors", "—"),
            "Растворимость": properties.get("solubility", "—"),
        }

        advanced_props = {
            "Оценка токсичности": properties.get("toxicity_estimate", "—"),
        }

        # Объединяем все свойства
        all_props = {}
        all_props.update(basic_props)
        all_props.update(physchem_props)
        all_props.update(advanced_props)

        df_data = {
            "Свойство": list(all_props.keys()),
            "Значение": list(all_props.values())
        }

        return pd.DataFrame(df_data)

    def create_isotopic_chart(self, isotopic_data: Dict[str, float]) -> Optional[go.Figure]:
        """Создание диаграммы изотопного распределения"""
        if "error" in isotopic_data:
            return None

        isotopes = list(isotopic_data.keys())
        abundances = list(isotopic_data.values())

        if not isotopes:
            return None

        fig = go.Figure(data=[
            go.Bar(
                x=isotopes,
                y=abundances,
                text=[f'{ab:.3f}' for ab in abundances],
                textposition='auto',
                marker_color='#1f77b4'
            )
        ])

        fig.update_layout(
            title="Основные изотопы молекулы",
            xaxis_title="Изотоп",
            yaxis_title="Количество атомов",
            height=300
        )

        return fig


def render_calculator_interface():
    """Отобразить интерфейс калькулятора"""
    st.header("🧮 Калькулятор молекулярных свойств")

    calculator = MolecularCalculator()

    # Ввод SMILES
    st.subheader("📝 Введите SMILES строку")

    col1, col2 = st.columns([3, 1])

    with col1:
        smiles_input = st.text_input(
            "SMILES:",
            placeholder="Например: CC(=O)O (уксусная кислота) или C1CCCCC1 (циклогексан)",
            help="Введите SMILES строку молекулы для расчета свойств"
        )

    with col2:
        if st.button("🔄 Рассчитать", width='stretch', type="primary"):
            if smiles_input:
                st.session_state.calculator_smiles = smiles_input
                st.session_state.show_calculation = True

    # Примеры SMILES
    with st.expander("💡 Примеры SMILES строк"):
        examples = {
            "Вода": "O",
            "Уксусная кислота": "CC(=O)O",
            "Глюкоза": "C(C1C(C(C(C(O1)O)O)O)O)O",
            "Аспирин": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Кофеин": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Бензол": "C1=CC=CC=C1",
            "Этанол": "CCO"
        }

        cols = st.columns(2)
        for i, (name, smiles) in enumerate(examples.items()):
            with cols[i % 2]:
                if st.button(f"{name}: {smiles}", key=f"example_{i}"):
                    st.session_state.calculator_smiles = smiles
                    st.session_state.show_calculation = True
                    st.rerun()

    # Расчет и отображение результатов
    if st.session_state.get('show_calculation', False):
        smiles_to_calculate = st.session_state.get('calculator_smiles', smiles_input)

        if smiles_to_calculate:
            st.divider()

            with st.spinner("Выполняю расчеты..."):
                properties = calculator.calculate_basic_properties(smiles_to_calculate)

            if "error" in properties:
                st.error(f"❌ {properties['error']}")
            else:
                # Основные свойства
                st.subheader("📊 Основные свойства")

                props_table = calculator.create_properties_table(properties)
                st.dataframe(
                    props_table,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "Свойство": st.column_config.TextColumn(width="medium"),
                        "Значение": st.column_config.TextColumn(width="large")
                    }
                )

                # 2D структура (если RDKit доступен)
                if RDKIT_AVAILABLE:
                    try:
                        mol = Chem.MolFromSmiles(smiles_to_calculate)
                        if mol:
                            st.subheader("🖼️ Структура молекулы")
                            img_col, info_col = st.columns([1, 2])

                            with img_col:
                                # Создаем изображение молекулы
                                from rdkit.Chem import Draw
                                img = Draw.MolToImage(mol, size=(300, 300))
                                st.image(img, caption=f"2D структура: {properties.get('formula', '')}")

                            with info_col:
                                st.metric("Формула", properties.get("formula", "—"))
                                st.metric("Масса", f"{properties.get('molecular_weight', '—')} Da")
                                st.metric("LogP", properties.get("logp", "—"))

                    except Exception as e:
                        st.warning(f"Не удалось создать изображение структуры: {str(e)}")

                # Изотопный анализ
                if "isotopic_pattern" in properties and isinstance(properties["isotopic_pattern"], dict):
                    st.subheader("🔬 Изотопный анализ")
                    isotopic_chart = calculator.create_isotopic_chart(properties["isotopic_pattern"])

                    if isotopic_chart:
                        st.plotly_chart(isotopic_chart, width='stretch')
                    else:
                        st.info("Изотопный анализ не доступен для этой молекулы")

                # Экспорт результатов
                st.subheader("📥 Экспорт результатов")

                col1, col2 = st.columns(2)

                with col1:
                    # Экспорт в CSV
                    csv_data = props_table.to_csv(index=False)
                    st.download_button(
                        label="📊 Скачать свойства (CSV)",
                        data=csv_data,
                        file_name=f"molecular_properties_{properties.get('formula', 'molecule')}.csv",
                        mime="text/csv",
                        width='stretch'
                    )

                with col2:
                    # Экспорт в JSON
                    import json
                    json_data = json.dumps(properties, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📋 Скачать свойства (JSON)",
                        data=json_data,
                        file_name=f"molecular_properties_{properties.get('formula', 'molecule')}.json",
                        mime="application/json",
                        width='stretch'
                    )

    # Информация о калькуляторе
    with st.expander("ℹ️ О калькуляторе"):
        st.markdown("""
        **🧮 Калькулятор молекулярных свойств** рассчитывает:

        - **Основные свойства**: формула, масса, количество атомов
        - **Физико-химические свойства**: LogP, TPSA, растворимость
        - **Структурные характеристики**: кольца, водородные связи
        - **Изотопный анализ**: распределение изотопов
        - **Оценка токсичности**: на основе правила пяти Липинского

        **Требования:** Для полных расчетов требуется установка RDKit.

        **Примеры SMILES:**
        - `CC(=O)O` - уксусная кислота
        - `C1CCCCC1` - циклогексан
        - `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` - кофеин
        """)
