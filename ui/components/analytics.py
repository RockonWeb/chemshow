"""
Компонент аналитики и дашбордов
Расширенная аналитика данных с KPI и визуализациями
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import sqlite3
import os

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


class AnalyticsDashboard:
    """Класс для создания аналитических дашбордов"""

    def __init__(self):
        self.database_paths = {}

    def load_database_stats(self) -> Dict[str, Any]:
        """Загрузка статистики из всех баз данных"""
        try:
            # Try absolute import first
            from config.settings import DATABASE_PATHS
        except ImportError:
            # Fallback to relative import
            from ..config.settings import DATABASE_PATHS

            stats = {
                "total_compounds": 0,
                "compounds_by_type": {},
                "mass_distribution": {"ranges": [], "counts": []},
                "organism_distribution": {},
                "class_distribution": {},
                "database_info": {}
            }

            for db_type, db_path in DATABASE_PATHS.items():
                if os.path.exists(db_path):
                    try:
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()

                        # Получаем общее количество соединений
                        cursor.execute(f"SELECT COUNT(*) FROM {db_type}")
                        count = cursor.fetchone()[0]
                        stats["compounds_by_type"][db_type] = count
                        stats["total_compounds"] += count

                        # Статистика по массам
                        if db_type in ['metabolites', 'carbohydrates', 'lipids']:
                            cursor.execute(f"SELECT exact_mass FROM {db_type} WHERE exact_mass IS NOT NULL")
                            masses = [row[0] for row in cursor.fetchall()]

                            if masses:
                                # Распределение по диапазонам масс
                                ranges = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 10000)]
                                for min_mass, max_mass in ranges:
                                    range_count = len([m for m in masses if min_mass <= m < max_mass])
                                    range_key = f"{min_mass}-{max_mass} Da"
                                    if range_key not in stats["mass_distribution"]["ranges"]:
                                        stats["mass_distribution"]["ranges"].append(range_key)
                                        stats["mass_distribution"]["counts"].append(0)
                                    range_idx = stats["mass_distribution"]["ranges"].index(range_key)
                                    stats["mass_distribution"]["counts"][range_idx] += range_count

                        # Статистика по организмам
                        if 'organism' in self._get_table_columns(cursor, db_type):
                            cursor.execute(f"SELECT organism, COUNT(*) FROM {db_type} WHERE organism IS NOT NULL GROUP BY organism ORDER BY COUNT(*) DESC LIMIT 10")
                            organism_data = cursor.fetchall()
                            for org, count in organism_data:
                                if org not in stats["organism_distribution"]:
                                    stats["organism_distribution"][org] = 0
                                stats["organism_distribution"][org] += count

                        # Статистика по классам (для метаболитов)
                        if db_type == 'metabolites' and 'class_name' in self._get_table_columns(cursor, db_type):
                            cursor.execute("SELECT class_name, COUNT(*) FROM metabolites WHERE class_name IS NOT NULL GROUP BY class_name ORDER BY COUNT(*) DESC LIMIT 10")
                            class_data = cursor.fetchall()
                            for class_name, count in class_data:
                                stats["class_distribution"][class_name] = count

                        # Информация о базе данных
                        stats["database_info"][db_type] = {
                            "path": db_path,
                            "size_mb": round(os.path.getsize(db_path) / (1024 * 1024), 2),
                            "last_modified": datetime.fromtimestamp(os.path.getmtime(db_path)).strftime("%Y-%m-%d %H:%M")
                        }

                        conn.close()

                    except Exception as e:
                        logger.error(f"Ошибка загрузки статистики для {db_type}: {e}")
                        stats["database_info"][db_type] = {"error": str(e)}

            return stats

        except Exception as e:
            logger.error(f"Ошибка загрузки статистики: {e}")
            return {"error": str(e)}

    def _get_table_columns(self, cursor, table_name: str) -> List[str]:
        """Получение списка колонок таблицы"""
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cursor.fetchall()]
        except:
            return []

    def create_overview_kpi(self, stats: Dict[str, Any]) -> None:
        """Создание общих KPI метрик"""
        if "error" in stats:
            st.error(f"❌ Ошибка загрузки статистики: {stats['error']}")
            return

        st.subheader("📊 Общая статистика")

        # Основные метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Всего соединений",
                f"{stats.get('total_compounds', 0):,}",
                help="Общее количество соединений во всех базах данных"
            )

        with col2:
            db_count = len([db for db in stats.get("database_info", {}) if "error" not in stats["database_info"][db]])
            st.metric(
                "Баз данных",
                db_count,
                help="Количество активных баз данных"
            )

        with col3:
            # Расчет общего размера баз данных
            total_size = sum([db_info.get("size_mb", 0) for db_info in stats.get("database_info", {}).values() if isinstance(db_info, dict) and "size_mb" in db_info])
            st.metric(
                "Общий размер",
                f"{total_size:.1f} MB",
                help="Суммарный размер всех баз данных"
            )

        with col4:
            avg_mass_range = "—"
            if stats.get("mass_distribution", {}).get("counts"):
                total_with_mass = sum(stats["mass_distribution"]["counts"])
                if total_with_mass > 0:
                    avg_mass_range = f"{total_with_mass:,}"
            st.metric(
                "С данными масс",
                avg_mass_range,
                help="Соединений с данными о молекулярной массе"
            )

    def create_compounds_distribution_chart(self, stats: Dict[str, Any]) -> Optional[go.Figure]:
        """Диаграмма распределения соединений по типам"""
        compounds_by_type = stats.get("compounds_by_type", {})

        if not compounds_by_type:
            return None

        compound_types = {
            "metabolites": "🧬 Метаболиты",
            "enzymes": "🧪 Ферменты",
            "proteins": "🔬 Белки",
            "carbohydrates": "🌾 Углеводы",
            "lipids": "🫧 Липиды"
        }

        labels = []
        values = []
        colors = []

        color_map = {
            "metabolites": "#1f77b4",
            "enzymes": "#ff7f0e",
            "proteins": "#2ca02c",
            "carbohydrates": "#d62728",
            "lipids": "#9467bd"
        }

        for db_type, count in compounds_by_type.items():
            labels.append(compound_types.get(db_type, db_type))
            values.append(count)
            colors.append(color_map.get(db_type, "#7f7f7f"))

        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo='label+percent',
                textposition='inside',
                hovertemplate='<b>%{label}</b><br>Количество: %{value}<br>Доля: %{percent}<extra></extra>'
            )
        ])

        fig.update_layout(
            title="Распределение соединений по типам",
            showlegend=False,
            height=400
        )

        return fig

    def create_mass_distribution_chart(self, stats: Dict[str, Any]) -> Optional[go.Figure]:
        """Диаграмма распределения по молекулярным массам"""
        mass_dist = stats.get("mass_distribution", {})

        if not mass_dist.get("ranges") or not mass_dist.get("counts"):
            return None

        # Фильтруем нулевые значения для лучшей визуализации
        filtered_data = [(r, c) for r, c in zip(mass_dist["ranges"], mass_dist["counts"]) if c > 0]

        if not filtered_data:
            return None

        ranges, counts = zip(*filtered_data)

        fig = go.Figure(data=[
            go.Bar(
                x=ranges,
                y=counts,
                marker_color='#1f77b4',
                text=counts,
                textposition='auto',
                hovertemplate='<b>Диапазон:</b> %{x}<br><b>Количество:</b> %{y}<extra></extra>'
            )
        ])

        fig.update_layout(
            title="Распределение соединений по молекулярной массе",
            xaxis_title="Диапазон массы (Da)",
            yaxis_title="Количество соединений",
            height=400
        )

        return fig

    def create_organism_distribution_chart(self, stats: Dict[str, Any]) -> Optional[go.Figure]:
        """Диаграмма распределения по организмам"""
        org_dist = stats.get("organism_distribution", {})

        if not org_dist:
            return None

        # Берем топ-10 организмов
        sorted_orgs = sorted(org_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        organisms, counts = zip(*sorted_orgs)

        fig = go.Figure(data=[
            go.Bar(
                x=organisms,
                y=counts,
                marker_color='#2ca02c',
                text=counts,
                textposition='auto',
                hovertemplate='<b>Организм:</b> %{x}<br><b>Количество:</b> %{y}<extra></extra>'
            )
        ])

        fig.update_layout(
            title="Топ-10 организмов по количеству соединений",
            xaxis_title="Организм",
            yaxis_title="Количество соединений",
            height=400,
            xaxis_tickangle=-45
        )

        return fig

    def create_class_distribution_chart(self, stats: Dict[str, Any]) -> Optional[go.Figure]:
        """Диаграмма распределения метаболитов по классам"""
        class_dist = stats.get("class_distribution", {})

        if not class_dist:
            return None

        # Берем топ-10 классов
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        classes, counts = zip(*sorted_classes)

        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=counts,
                marker_color='#ff7f0e',
                text=counts,
                textposition='auto',
                hovertemplate='<b>Класс:</b> %{x}<br><b>Количество:</b> %{y}<extra></extra>'
            )
        ])

        fig.update_layout(
            title="Топ-10 классов метаболитов",
            xaxis_title="Класс соединений",
            yaxis_title="Количество",
            height=400,
            xaxis_tickangle=-45
        )

        return fig

    def create_database_info_table(self, stats: Dict[str, Any]) -> pd.DataFrame:
        """Создание таблицы информации о базах данных"""
        db_info = stats.get("database_info", {})

        if not db_info:
            return pd.DataFrame()

        table_data = []
        for db_type, info in db_info.items():
            if isinstance(info, dict) and "error" not in info:
                table_data.append({
                    "База данных": db_type.title(),
                    "Размер (MB)": info.get("size_mb", "—"),
                    "Последнее изменение": info.get("last_modified", "—"),
                    "Статус": "✅ Активна"
                })
            else:
                table_data.append({
                    "База данных": db_type.title(),
                    "Размер (MB)": "—",
                    "Последнее изменение": "—",
                    "Статус": "❌ Ошибка"
                })

        return pd.DataFrame(table_data)


def render_analytics_dashboard():
    """Отобразить дашборд аналитики"""
    st.header("📊 Аналитика и статистика")

    dashboard = AnalyticsDashboard()

    # Кнопка обновления данных
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Обновить данные", width='stretch'):
            st.cache_data.clear()
            st.rerun()

    # Загружаем статистику с кешированием
    @st.cache_data(ttl=3600)  # Кеш на 1 час
    def load_stats():
        return dashboard.load_database_stats()

    with st.spinner("Загружаю статистику..."):
        stats = load_stats()

    # KPI метрики
    dashboard.create_overview_kpi(stats)

    st.divider()

    # Визуализации
    st.subheader("📈 Визуализации")

    # Создаем вкладки для разных типов диаграмм
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Распределение", "⚖️ Массы", "🦠 Организмы", "🏷️ Классы"])

    with tab1:
        chart = dashboard.create_compounds_distribution_chart(stats)
        if chart:
            st.plotly_chart(chart, width='stretch')
        else:
            st.info("Данные для диаграммы распределения недоступны")

    with tab2:
        chart = dashboard.create_mass_distribution_chart(stats)
        if chart:
            st.plotly_chart(chart, width='stretch')
        else:
            st.info("Данные о молекулярных массах недоступны")

    with tab3:
        chart = dashboard.create_organism_distribution_chart(stats)
        if chart:
            st.plotly_chart(chart, width='stretch')
        else:
            st.info("Данные об организмах недоступны")

    with tab4:
        chart = dashboard.create_class_distribution_chart(stats)
        if chart:
            st.plotly_chart(chart, width='stretch')
        else:
            st.info("Данные о классах метаболитов недоступны")

    # Таблица информации о базах данных
    st.subheader("🗄️ Информация о базах данных")

    db_table = dashboard.create_database_info_table(stats)
    if not db_table.empty:
        st.dataframe(
            db_table,
            width='stretch',
            hide_index=True,
            column_config={
                "База данных": st.column_config.TextColumn(width="medium"),
                "Размер (MB)": st.column_config.NumberColumn(width="small", format="%.2f"),
                "Последнее изменение": st.column_config.TextColumn(width="medium"),
                "Статус": st.column_config.TextColumn(width="small")
            }
        )

        # Экспорт статистики
        st.subheader("📥 Экспорт статистики")

        col1, col2 = st.columns(2)

        with col1:
            # Экспорт в CSV
            csv_data = db_table.to_csv(index=False)
            st.download_button(
                label="📊 Скачать статистику (CSV)",
                data=csv_data,
                file_name="database_statistics.csv",
                mime="text/csv",
                width='stretch'
            )

        with col2:
            # Экспорт в JSON
            import json
            json_data = json.dumps(stats, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="📋 Скачать статистику (JSON)",
                data=json_data,
                file_name="database_statistics.json",
                mime="application/json",
                width='stretch'
            )
    else:
        st.warning("Информация о базах данных недоступна")

    # Дополнительная информация
    with st.expander("ℹ️ О аналитике"):
        st.markdown("""
        **📊 Дашборд аналитики** предоставляет:

        - **Общую статистику**: количество соединений, размер баз данных
        - **Распределение по типам**: соотношение метаболитов, ферментов, белков и т.д.
        - **Анализ масс**: распределение соединений по диапазонам молекулярных масс
        - **Организмы**: топ-10 организмов с наибольшим количеством соединений
        - **Классы**: распределение метаболитов по химическим классам

        **Обновление данных**: Статистика кешируется на 1 час для производительности.
        Нажмите "🔄 Обновить данные" для принудительного обновления.
        """)
