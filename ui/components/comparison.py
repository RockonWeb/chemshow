"""
Компонент сравнительного анализа соединений
Позволяет сравнивать несколько соединений side-by-side
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import logging

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


class CompoundComparator:
    """Класс для сравнения соединений"""

    def __init__(self):
        self.selected_compounds = []
        self.comparison_data = {}

    def add_compound(self, compound: Dict[str, Any], compound_type: str):
        """Добавить соединение для сравнения"""
        compound_id = f"{compound_type}_{compound.get('id', hash(str(compound)))}"
        compound_name = get_display_name(compound)

        if compound_id not in [c['id'] for c in self.selected_compounds]:
            self.selected_compounds.append({
                'id': compound_id,
                'name': compound_name,
                'type': compound_type,
                'data': compound
            })

            # Ограничиваем до 5 соединений для лучшей производительности
            if len(self.selected_compounds) > 5:
                self.selected_compounds.pop(0)

    def remove_compound(self, compound_id: str):
        """Удалить соединение из сравнения"""
        self.selected_compounds = [c for c in self.selected_compounds if c['id'] != compound_id]

    def clear_all(self):
        """Очистить все соединения"""
        self.selected_compounds = []

    def get_comparison_table(self) -> pd.DataFrame:
        """Создать таблицу сравнения свойств"""
        if not self.selected_compounds:
            return pd.DataFrame()

        comparison_rows = []
        properties = []

        # Собираем все возможные свойства
        for compound in self.selected_compounds:
            compound_data = compound['data']
            compound_type = compound['type']

            row = {'Название': compound['name'], 'Тип': compound_type}

            # Общие свойства
            if 'formula' in compound_data:
                row['Формула'] = safe_get_value(compound_data, 'formula', '—')
            if 'exact_mass' in compound_data:
                row['Масса (Da)'] = format_mass(safe_get_value(compound_data, 'exact_mass'), '')
            if 'organism' in compound_data:
                row['Организм'] = safe_get_value(compound_data, 'organism', '—')

            # Специфические свойства по типам
            if compound_type == 'metabolites':
                if 'class_name' in compound_data:
                    row['Класс'] = safe_get_value(compound_data, 'class_name', '—')
                if 'pathway' in compound_data:
                    row['Путь'] = safe_get_value(compound_data, 'pathway', '—')

            elif compound_type == 'enzymes':
                if 'ec_number' in compound_data:
                    row['EC номер'] = safe_get_value(compound_data, 'ec_number', '—')
                if 'family' in compound_data:
                    row['Семейство'] = safe_get_value(compound_data, 'family', '—')
                if 'reaction' in compound_data:
                    row['Реакция'] = safe_get_value(compound_data, 'reaction', '—')

            elif compound_type == 'proteins':
                if 'function' in compound_data:
                    row['Функция'] = safe_get_value(compound_data, 'function', '—')
                if 'family' in compound_data:
                    row['Семейство'] = safe_get_value(compound_data, 'family', '—')
                if 'sequence_length' in compound_data:
                    row['Длина последовательности'] = safe_get_value(compound_data, 'sequence_length', '—')

            elif compound_type == 'carbohydrates':
                if 'type' in compound_data:
                    row['Тип'] = safe_get_value(compound_data, 'type', '—')
                if 'degree_polymerization' in compound_data:
                    row['Степень полимеризации'] = safe_get_value(compound_data, 'degree_polymerization', '—')

            elif compound_type == 'lipids':
                if 'type' in compound_data:
                    row['Тип'] = safe_get_value(compound_data, 'type', '—')
                if 'fatty_acids' in compound_data:
                    row['Жирные кислоты'] = safe_get_value(compound_data, 'fatty_acids', '—')

            comparison_rows.append(row)

        if comparison_rows:
            df = pd.DataFrame(comparison_rows)
            # Заполняем отсутствующие значения
            df = df.fillna('—')
            return df

        return pd.DataFrame()

    def create_mass_comparison_chart(self) -> Optional[go.Figure]:
        """Создать диаграмму сравнения масс"""
        mass_data = []
        names = []

        for compound in self.selected_compounds:
            mass = safe_get_value(compound['data'], 'exact_mass')
            if mass and isinstance(mass, (int, float)):
                mass_data.append(mass)
                names.append(compound['name'])

        if len(mass_data) >= 2:
            fig = go.Figure(data=[
                go.Bar(
                    x=names,
                    y=mass_data,
                    text=[f'{m:.2f}' for m in mass_data],
                    textposition='auto',
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(names)]
                )
            ])

            fig.update_layout(
                title="Сравнение молекулярных масс",
                xaxis_title="Соединение",
                yaxis_title="Масса (Da)",
                height=400
            )

            return fig

        return None

    def create_property_radar_chart(self) -> Optional[go.Figure]:
        """Создать радар-диаграмму свойств"""
        # Для простоты используем только числовые свойства
        numeric_props = {}

        for compound in self.selected_compounds:
            compound_data = compound['data']
            name = compound['name']

            # Собираем числовые свойства
            props = {}
            if 'exact_mass' in compound_data and isinstance(compound_data['exact_mass'], (int, float)):
                props['Масса'] = compound_data['exact_mass']

            # Для ферментов - добавим другие метрики если есть
            if 'k_cat' in compound_data and isinstance(compound_data['k_cat'], (int, float)):
                props['k_cat'] = compound_data['k_cat']

            if 'km' in compound_data and isinstance(compound_data['km'], (int, float)):
                props['Km'] = compound_data['km']

            if props:
                numeric_props[name] = props

        if len(numeric_props) >= 2:
            # Находим общие свойства
            all_props = set()
            for props in numeric_props.values():
                all_props.update(props.keys())

            if len(all_props) >= 2:
                fig = go.Figure()

                for name, props in numeric_props.items():
                    # Нормализуем значения для радар-диаграммы
                    values = []
                    for prop in all_props:
                        if prop in props:
                            values.append(props[prop])
                        else:
                            values.append(0)

                    # Нормализация
                    if values:
                        max_val = max(values) if max(values) > 0 else 1
                        values = [v/max_val for v in values]

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=list(all_props),
                        fill='toself',
                        name=name
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Сравнение свойств соединений",
                    height=500
                )

                return fig

        return None


def render_comparison_interface(comparator: CompoundComparator):
    """Отобразить интерфейс сравнительного анализа"""
    st.header("🔬 Сравнение соединений")

    # Управление выбранными соединениями
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if comparator.selected_compounds:
            st.subheader(f"Выбрано для сравнения: {len(comparator.selected_compounds)}")

            # Список выбранных соединений
            for compound in comparator.selected_compounds:
                cols = st.columns([3, 1])
                with cols[0]:
                    st.write(f"**{compound['name']}** ({compound['type']})")
                with cols[1]:
                    if st.button("❌", key=f"remove_{compound['id']}",
                               help=f"Удалить {compound['name']}"):
                        comparator.remove_compound(compound['id'])
                        st.rerun()

    with col2:
        if comparator.selected_compounds:
            if st.button("🗑️ Очистить все", width='stretch'):
                comparator.clear_all()
                st.rerun()

    with col3:
        if len(comparator.selected_compounds) >= 2:
            if st.button("📊 Показать сравнение", width='stretch', type="primary"):
                st.session_state.show_comparison = True

    # Показать сравнение если выбрано достаточно соединений
    if st.session_state.get('show_comparison', False) and len(comparator.selected_compounds) >= 2:

        st.divider()

        # Таблица сравнения
        st.subheader("📋 Таблица сравнения")

        comparison_df = comparator.get_comparison_table()
        if not comparison_df.empty:
            st.dataframe(
                comparison_df,
                width='stretch',
                hide_index=True
            )

            # Скачивание таблицы
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="📥 Скачать как CSV",
                data=csv,
                file_name="compound_comparison.csv",
                mime="text/csv",
                width='stretch'
            )

        # Диаграммы сравнения
        st.subheader("📊 Визуализация сравнения")

        col1, col2 = st.columns(2)

        with col1:
            mass_chart = comparator.create_mass_comparison_chart()
            if mass_chart:
                st.plotly_chart(mass_chart, width='stretch')

        with col2:
            radar_chart = comparator.create_property_radar_chart()
            if radar_chart:
                st.plotly_chart(radar_chart, width='stretch')

        # Кнопка закрытия
        if st.button("🔙 Вернуться к выбору", width='stretch'):
            st.session_state.show_comparison = False
            st.rerun()

    elif len(comparator.selected_compounds) < 2:
        st.info("👆 Выберите минимум 2 соединения для сравнения")
        st.session_state.show_comparison = False


def add_to_comparison_button(compound: Dict[str, Any], compound_type: str, comparator: CompoundComparator):
    """Кнопка добавления соединения в сравнение"""
    compound_name = get_display_name(compound)
    compound_id = f"{compound_type}_{compound.get('id', hash(str(compound)))}"

    # Проверяем, уже ли добавлено
    is_added = compound_id in [c['id'] for c in comparator.selected_compounds]

    if is_added:
        if st.button("✅ В сравнении", key=f"compare_{compound_id}", disabled=True):
            pass
    else:
        if st.button("🔍 Добавить к сравнению", key=f"compare_{compound_id}"):
            comparator.add_compound(compound, compound_type)
            st.success(f"✅ {compound_name} добавлен к сравнению")
            st.rerun()


# Глобальный экземпляр компаратора
comparison_comparator = CompoundComparator()
