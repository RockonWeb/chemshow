"""
Рефакторингованное основное приложение Справочник соединений
Использует модульную архитектуру для лучшей поддерживаемости
"""
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь для корректных импортов
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path

# Импорты из созданных модулей
import sys
from pathlib import Path

# Добавляем пути для корректных импортов
config_dir = Path(__file__).parent / "config"
services_dir = Path(__file__).parent / "services"
components_dir = Path(__file__).parent / "components"

sys.path.insert(0, str(config_dir))
sys.path.insert(0, str(services_dir))
sys.path.insert(0, str(components_dir))

from config.settings import (
    UI_CONFIG, LOGGING_CONFIG, get_database_paths,
    SEARCH_CONFIG, SEARCH_PRESETS
)
from services.database import get_database_stats
from services.search_service import search_service
from components.styles import inject_styles
from components.utils import create_stats_html, format_search_query
from components.search_form import (
    render_search_form, handle_search_form,
    render_pagination, render_view_toggle,
    render_results_header, render_close_details_buttons
)
from components.cards import (
    render_metabolite_card, render_enzyme_card,
    render_protein_card, render_carbohydrate_card,
    render_lipid_card
)
from components.details import (
    show_metabolite_details, show_enzyme_details,
    show_protein_details, show_carbohydrate_details,
    show_lipid_details
)
from components.visualization_3d import (
    render_3d_structure, display_molecule_properties,
    check_dependencies, install_instructions
)

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)

# -------------------------
# Основная логика приложения
# -------------------------

def initialize_session_state():
    """Инициализация состояния сессии"""
    defaults = {
        "page": 1,
        "page_size": UI_CONFIG["default_page_size"],
        "search_submitted": False,
        "view_mode": "Карточки",
        "show_metabolite_details": False,
        "show_enzyme_details": False,
        "show_protein_details": False,
        "show_carbohydrate_details": False,
        "show_lipid_details": False,
        "search_results": {},
        "selected_metabolite": None,
        "selected_enzyme": None,
        "selected_protein": None,
        "selected_carbohydrate": None,
        "selected_lipid": None,
        "last_query": "",
        "last_mass": None,
        "last_organism_type": "Все",
        "last_tolerance_ppm": 1000,
        "show_3d_structure": False,
        "show_molecule_properties": False,
        "modal_structure_smiles": None,
        "modal_structure_action": None,
        "current_smiles": None,
        "current_molecule_name": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def close_all_dialogs():
    """Безопасно закрывает все диалоги"""
    st.session_state.show_metabolite_details = False
    st.session_state.show_enzyme_details = False
    st.session_state.show_protein_details = False
    st.session_state.show_carbohydrate_details = False
    st.session_state.show_lipid_details = False
    st.session_state.selected_metabolite = None
    st.session_state.selected_enzyme = None
    st.session_state.selected_protein = None
    st.session_state.selected_carbohydrate = None
    st.session_state.selected_lipid = None


def open_dialog_safely(dialog_type: str, entity: Dict[str, Any]):
    """Безопасно открывает диалог, закрывая все остальные"""
    # Сначала закрываем все диалоги
    close_all_dialogs()
    
    # Затем открываем нужный диалог
    if dialog_type == "metabolite":
        st.session_state.selected_metabolite = entity
        st.session_state.show_metabolite_details = True
    elif dialog_type == "enzyme":
        st.session_state.selected_enzyme = entity
        st.session_state.show_enzyme_details = True
    elif dialog_type == "protein":
        st.session_state.selected_protein = entity
        st.session_state.show_protein_details = True
    elif dialog_type == "carbohydrate":
        st.session_state.selected_carbohydrate = entity
        st.session_state.show_carbohydrate_details = True
    elif dialog_type == "lipid":
        st.session_state.selected_lipid = entity
        st.session_state.show_lipid_details = True


def render_database_stats():
    """Отображение статистики базы данных"""
    try:
        totals = get_database_stats()
        stats_html = create_stats_html(totals)
        st.markdown(stats_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Ошибка получения статистики БД: {e}")
        st.error("❌ Ошибка получения статистики базы данных")
        return False
    return True


def handle_search_and_display():
    """Обработка поиска и отображение результатов"""
    # Поисковая форма
    form_data = render_search_form()
    search_params = handle_search_form(form_data)

    # Выполнение поиска
    if search_params.get("search_submitted"):
        with st.status("Выполняется поиск...", expanded=False):
            try:
                results = search_service.unified_search(
                    query=search_params.get("query"),
                    mass=search_params.get("mass"),
                    tolerance_ppm=search_params.get("tolerance_ppm", 1000),
                    organism_type=search_params.get("organism_type", "Все"),
                    page=st.session_state.page,
                    page_size=st.session_state.page_size
                )

                # Сохранение результатов
                st.session_state.search_results = results
                st.session_state.last_query = search_params.get("query", "")
                st.session_state.last_mass = search_params.get("mass")
                st.session_state.last_organism_type = search_params.get("organism_type", "Все")
                st.session_state.last_tolerance_ppm = search_params.get("tolerance_ppm", 1000)
                st.session_state.search_submitted = True

            except Exception as e:
                logger.error(f"Ошибка поиска: {e}")
                st.error(f"❌ Ошибка выполнения поиска: {str(e)}")
                return

        st.rerun()

    # Отображение результатов
    if st.session_state.get("search_submitted") and st.session_state.get("search_results"):
        display_search_results()


def display_search_results():
    """Отображение результатов поиска"""
    results = st.session_state.get("search_results", {})

    # Заголовок и статистика
    search_totals = search_service.get_search_totals(results)
    total_all = sum(search_totals.values())

    if total_all > 0:
        render_results_header(total_all, {
            "query": st.session_state.get("last_query", ""),
            "mass": st.session_state.get("last_mass")
        })

        # Переключатель вида
        new_view = render_view_toggle(st.session_state.view_mode)
        if new_view:
            st.session_state.view_mode = new_view

        # Отображение результатов по типам
        display_entity_results(results, "metabolites", "🧬 Метаболиты",
                              render_metabolite_card, show_metabolite_details)
        display_entity_results(results, "enzymes", "🧪 Ферменты",
                              render_enzyme_card, show_enzyme_details)
        display_entity_results(results, "proteins", "🔬 Белки",
                              render_protein_card, show_protein_details)
        display_entity_results(results, "carbohydrates", "🌾 Углеводы",
                              render_carbohydrate_card, show_carbohydrate_details)
        display_entity_results(results, "lipids", "🫧 Липиды",
                              render_lipid_card, show_lipid_details)

        # Пагинация
        max_results = search_service.get_max_results_for_pagination(search_totals)
        if max_results > st.session_state.page_size:
            new_page = render_pagination(
                st.session_state.page,
                (max_results + st.session_state.page_size - 1) // st.session_state.page_size,
                st.session_state.page_size,
                max_results
            )
            if new_page:
                st.session_state.page = new_page
                # Повторный поиск с новой страницей
                results = search_service.unified_search(
                    query=st.session_state.get("last_query", ""),
                    mass=st.session_state.get("last_mass"),
                    tolerance_ppm=st.session_state.get("last_tolerance_ppm", 1000),
                    organism_type=st.session_state.get("last_organism_type", "Все"),
                    page=st.session_state.page,
                    page_size=st.session_state.page_size
                )
                st.session_state.search_results = results
                st.rerun()
    else:
        st.warning("🔍 Результаты не найдены. Попробуйте изменить параметры поиска.")


def display_entity_results(results: Dict, entity_type: str, header: str,
                          card_renderer, details_renderer):
    """Отображение результатов для конкретного типа сущностей"""
    entities = results.get(entity_type, {}).get("data", [])
    if not entities:
        return

    st.subheader(f"{header} ({len(entities)})")

    if st.session_state.view_mode == "Таблица":
        # Табличный вид
        display_entity_table(entities, entity_type)
    else:
        # Карточный вид
        cols = st.columns(UI_CONFIG["cards_per_row"])
        for idx, entity in enumerate(entities):
            with cols[idx % UI_CONFIG["cards_per_row"]]:
                card_key = f"{entity_type[:-1]}_card_{entity.get('id', hash(str(entity)))}"
                card_renderer(entity, card_key)


def display_entity_table(entities: list, entity_type: str):
    """Отображение сущностей в табличном виде"""
    if not entities:
        return

    # Создание DataFrame в зависимости от типа
    rows = []
    for entity in entities:
        if entity_type == "metabolites":
            rows.append({
                "Название": entity.get("name_ru", "") or entity.get("name", ""),
                "Формула": entity.get("formula", ""),
                "Масса": float(entity["exact_mass"]) if entity.get("exact_mass") else None,
                "Класс": entity.get("class_name", ""),
            })
        elif entity_type == "enzymes":
            rows.append({
                "Название": entity.get("name_ru", "") or entity.get("name", ""),
                "EC номер": entity.get("ec_number", ""),
                "Организм": entity.get("organism", ""),
                "Семейство": entity.get("family", ""),
            })
        elif entity_type == "proteins":
            rows.append({
                "Название": entity.get("name_ru", "") or entity.get("name", ""),
                "Функция": entity.get("function", ""),
                "Организм": entity.get("organism", ""),
                "Семейство": entity.get("family", ""),
            })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
        st.info("💡 **Совет:** Переключитесь в режим 'Карточки' для просмотра деталей")


def handle_modal_signals():
    """Обработка сигналов от модальных окон"""
    # Проверяем сигналы от модальных окон
    if st.session_state.get("modal_structure_action") and st.session_state.get("modal_structure_smiles"):
        action = st.session_state.modal_structure_action
        smiles = st.session_state.modal_structure_smiles

        if action == "show_3d":
            st.session_state.show_3d_structure = True
            st.session_state.current_smiles = smiles
            st.session_state.current_molecule_name = "Структура из модального окна"
        elif action == "show_2d":
            st.session_state.show_molecule_properties = True
            st.session_state.current_smiles = smiles
            st.session_state.current_molecule_name = "Структура из модального окна"

        # Сбрасываем сигналы
        st.session_state.modal_structure_action = None
        st.session_state.modal_structure_smiles = None

    # Обработка сигналов от модальных окон через session_state
    # (более надежный способ, чем query_params)
    pass  # Пока оставляем пустым, так как основная обработка уже выше


def display_details():
    """Отображение детальной информации в модальных окнах"""

    # Сначала обрабатываем сигналы от модальных окон
    handle_modal_signals()

    # Проверяем, что только один диалог открыт одновременно
    dialog_states = [
        st.session_state.get("show_metabolite_details", False),
        st.session_state.get("show_enzyme_details", False),
        st.session_state.get("show_protein_details", False),
        st.session_state.get("show_carbohydrate_details", False),
        st.session_state.get("show_lipid_details", False)
    ]
    
    # Если открыто больше одного диалога, закрываем все кроме первого
    if sum(dialog_states) > 1:
        # Закрываем все диалоги
        st.session_state.show_metabolite_details = False
        st.session_state.show_enzyme_details = False
        st.session_state.show_protein_details = False
        st.session_state.show_carbohydrate_details = False
        st.session_state.show_lipid_details = False
        st.session_state.selected_metabolite = None
        st.session_state.selected_enzyme = None
        st.session_state.selected_protein = None
        st.session_state.selected_carbohydrate = None
        st.session_state.selected_lipid = None
        st.rerun()
        return

    # Метаболиты
    if st.session_state.get("show_metabolite_details") and st.session_state.get("selected_metabolite"):
        try:
            show_metabolite_details(st.session_state.selected_metabolite)
        except Exception as e:
            logger.error(f"Ошибка отображения деталей метаболита: {e}")
            st.session_state.show_metabolite_details = False
            st.session_state.selected_metabolite = None

    # Ферменты
    elif st.session_state.get("show_enzyme_details") and st.session_state.get("selected_enzyme"):
        try:
            show_enzyme_details(st.session_state.selected_enzyme)
        except Exception as e:
            logger.error(f"Ошибка отображения деталей фермента: {e}")
            st.session_state.show_enzyme_details = False
            st.session_state.selected_enzyme = None

    # Белки
    elif st.session_state.get("show_protein_details") and st.session_state.get("selected_protein"):
        try:
            show_protein_details(st.session_state.selected_protein)
        except Exception as e:
            logger.error(f"Ошибка отображения деталей белка: {e}")
            st.session_state.show_protein_details = False
            st.session_state.selected_protein = None

    # Углеводы
    elif st.session_state.get("show_carbohydrate_details") and st.session_state.get("selected_carbohydrate"):
        try:
            show_carbohydrate_details(st.session_state.selected_carbohydrate)
        except Exception as e:
            logger.error(f"Ошибка отображения деталей углевода: {e}")
            st.session_state.show_carbohydrate_details = False
            st.session_state.selected_carbohydrate = None

    # Липиды
    elif st.session_state.get("show_lipid_details") and st.session_state.get("selected_lipid"):
        try:
            show_lipid_details(st.session_state.selected_lipid)
        except Exception as e:
            logger.error(f"Ошибка отображения деталей липида: {e}")
            st.session_state.show_lipid_details = False
            st.session_state.selected_lipid = None


def handle_close_buttons():
    """Обработка кнопок закрытия деталей"""
    close_buttons = render_close_details_buttons()

    if close_buttons.get("close_metabolite"):
        close_all_dialogs()
        st.rerun()

    if close_buttons.get("close_enzyme"):
        close_all_dialogs()
        st.rerun()

    if close_buttons.get("close_protein"):
        close_all_dialogs()
        st.rerun()

    if close_buttons.get("close_carbohydrate"):
        close_all_dialogs()
        st.rerun()

    if close_buttons.get("close_lipid"):
        close_all_dialogs()
        st.rerun()


# -------------------------
# Основное приложение
# -------------------------

def main():
    """Главная функция приложения"""
    # Настройка страницы
    st.set_page_config(
        page_title=UI_CONFIG["page_title"],
        page_icon=UI_CONFIG["page_icon"],
        layout=UI_CONFIG["layout"]
    )

    # Инициализация
    initialize_session_state()
    inject_styles()

    # Проверка зависимостей для 3D-визуализации
    vis_deps = check_dependencies()
    if not all(vis_deps.values()):
        st.warning("⚠️ Некоторые зависимости для расширенных функций недоступны")

        with st.expander("📦 Статус зависимостей"):
            for dep, available in vis_deps.items():
                status = "✅ Доступно" if available else "❌ Отсутствует"
                st.write(f"**{dep}:** {status}")

            if not all(vis_deps.values()):
                st.markdown("### Инструкции по установке:")
                st.code(install_instructions(), language="bash")

    # Проверка баз данных
    try:
        get_database_paths()
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.markdown("""
        **Для решения запустите:**
        ```bash
        python data/create_all_databases.py
        ```
        """)
        st.stop()

    # Заголовок
    st.title(UI_CONFIG["page_title"])
    st.markdown("**Унифицированный поиск по метаболитам, ферментам, белкам, углеводам и липидам**",
                help="Поиск по всем типам соединений в одной форме")

    # Статистика базы данных
    st.markdown("### Статистика базы данных")

    # Отображение статистики базы данных
    if not render_database_stats():
        st.stop()

    # Поиск и результаты
    handle_search_and_display()

    # Детали
    display_details()
    handle_close_buttons()

    # Футер
    st.markdown("---")
    st.markdown("🧬 **Справочник** - поиск среди метаболитов, ферментов, белков, углеводов и липидов")


if __name__ == "__main__":
    main()
