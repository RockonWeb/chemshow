"""
Компоненты формы поиска
"""
import streamlit as st
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Добавляем путь к config для корректных импортов
config_dir = Path(__file__).parent.parent / "config"
sys.path.insert(0, str(config_dir))

from config.settings import (
    SEARCH_CONFIG, ORGANISM_TYPES, SEARCH_PRESETS
)
from .utils import validate_search_params


def render_search_form() -> Dict[str, Any]:
    """Отрисовывает форму поиска и возвращает введенные значения"""
    with st.form("unified_search_form"):
        st.subheader("🔍 Поиск")

        # Основное поле поиска
        search_query = st.text_input(
            "Поисковый запрос",
            placeholder="Например: глюкоза, dehydrogenase, insulin",
            help="Поиск по названию, формуле, EC номеру, функции. Запрос автоматически приводится к формату с заглавной буквы. Нажмите Enter для быстрого поиска.",
            key="search_query_input"
        )

        # Кнопка поиска
        search_submitted = st.form_submit_button("🔍 Найти", use_container_width=True, type="primary")

        # Дополнительные настройки
        with st.expander("⚙️ Дополнительные настройки", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                mass_query = st.number_input(
                    "Масса (m/z) для поиска соединений",
                    min_value=0.0,
                    step=0.001,
                    format="%.6f",
                    help="Поиск по массе среди метаболитов (Da), ферментов и белков (kDa). Оставьте 0 для поиска только по названию.",
                    key="mass_query_input"
                )

                tolerance_ppm = st.slider(
                    "Допуск (ppm)",
                    min_value=SEARCH_CONFIG["min_tolerance_ppm"],
                    max_value=SEARCH_CONFIG["max_tolerance_ppm"],
                    value=SEARCH_CONFIG["default_tolerance_ppm"],
                    step=50,
                    help="Частей на миллион. 250 ppm = ±0.025% от массы, 1000 ppm = ±0.1% от массы, 10000 ppm = ±1% от массы"
                )

            with col2:
                organism_type = st.selectbox(
                    "🌱 Тип организма",
                    ORGANISM_TYPES,
                    help="Фильтрация по типу организма"
                )

                page_size = st.selectbox(
                    "Размер страницы",
                    options=[SEARCH_CONFIG["min_page_size"],
                            50,
                            100,
                            SEARCH_CONFIG["max_page_size"]],
                    index=1,  # 50 по умолчанию
                    help="Количество результатов на странице"
                )

        # Адаптивные пресеты поиска
        st.caption("💡 Быстрые пресеты (нажмите для поиска):")
        # Используем адаптивное количество колонок
        presets_col1, presets_col2, presets_col3, presets_col4 = st.columns([1, 1, 1, 1])

        preset_buttons = {}

        with presets_col1:
            preset_buttons["glucose"] = st.form_submit_button(
                "Глюкоза",
                use_container_width=True
            )

        with presets_col2:
            preset_buttons["dehydrogenase"] = st.form_submit_button(
                "Dehydrogenase",
                use_container_width=True
            )

        with presets_col3:
            preset_buttons["formaldehyde"] = st.form_submit_button(
                "Formaldehyde",
                use_container_width=True
            )

        with presets_col4:
            preset_buttons["atp"] = st.form_submit_button(
                "ATP",
                use_container_width=True
            )

        # Определяем, какая кнопка была нажата
        active_preset = None
        for preset_name, was_pressed in preset_buttons.items():
            if was_pressed:
                active_preset = preset_name
                break

        # Возвращаем все значения формы
        return {
            "search_submitted": search_submitted,
            "search_query": search_query,
            "mass_query": mass_query,
            "tolerance_ppm": tolerance_ppm,
            "organism_type": organism_type,
            "page_size": page_size,
            "active_preset": active_preset
        }


def handle_search_form(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """Обрабатывает данные формы поиска"""
    search_submitted = form_data["search_submitted"]
    active_preset = form_data["active_preset"]

    # Если был выбран пресет, используем его значения
    if active_preset:
        preset_values = SEARCH_PRESETS[active_preset]
        query = preset_values
        mass = None
        # Конвертация tolerance_ppm из строки в число
        try:
            tolerance_ppm = int(SEARCH_CONFIG["default_tolerance_ppm"])
        except (ValueError, TypeError):
            tolerance_ppm = 1000
        organism_type = "Все"
        page_size = SEARCH_CONFIG["default_page_size"]
        search_submitted = True
    else:
        # Используем введенные значения
        query = form_data["search_query"]

        # Конвертация mass из строки в число
        try:
            mass_raw = form_data["mass_query"]
            if isinstance(mass_raw, str):
                mass_value = float(mass_raw) if mass_raw.strip() else 0.0
            else:
                mass_value = float(mass_raw) if mass_raw else 0.0
            mass = mass_value if mass_value > 0 else None
        except (ValueError, TypeError):
            mass = None

        # Конвертация tolerance_ppm из строки в число
        try:
            tolerance_ppm = int(form_data["tolerance_ppm"])
        except (ValueError, TypeError):
            tolerance_ppm = SEARCH_CONFIG["default_tolerance_ppm"]

        organism_type = form_data["organism_type"]
        page_size = form_data["page_size"]

    # Валидация параметров
    validation = validate_search_params(query, mass, tolerance_ppm)

    if not validation["valid"]:
        st.error("Ошибка в параметрах поиска:")
        for error in validation["errors"]:
            st.error(f"• {error}")
        return {}

    if validation["warnings"]:
        for warning in validation["warnings"]:
            st.warning(f"⚠️ {warning}")

    # Возвращаем обработанные параметры
    return {
        "query": query,
        "mass": mass,
        "tolerance_ppm": tolerance_ppm,
        "organism_type": organism_type,
        "page_size": page_size,
        "search_submitted": search_submitted
    }


def render_pagination(current_page: int, total_pages: int, page_size: int,
                     total_items: int) -> Optional[int]:
    """Отрисовывает элементы пагинации и возвращает новую страницу или None"""
    if total_pages <= 1:
        return None

    st.subheader("📄 Пагинация")

    col1, col2, col3 = st.columns([1, 2, 1])

    new_page = None

    with col1:
        if st.button("⬅️ Предыдущая", key="prev_page", disabled=current_page <= 1):
            new_page = max(1, current_page - 1)

    with col2:
        st.markdown(f"Страница {current_page} из {total_pages}")

        # Информация о показываемых элементах
        start_item = (current_page - 1) * page_size + 1
        end_item = min(current_page * page_size, total_items)
        st.caption(f"Показаны результаты {start_item}-{end_item} из {total_items}")

    with col3:
        if st.button("Следующая ➡️", key="next_page", disabled=current_page >= total_pages):
            new_page = min(total_pages, current_page + 1)

    return new_page


def render_view_toggle(current_view: str) -> Optional[str]:
    """Отрисовывает переключатель вида (Карточки/Таблица)"""
    view_choice = st.radio(
        "Вид",
        options=["Карточки", "Таблица"],
        horizontal=True,
        index=["Карточки", "Таблица"].index(current_view),
        key="view_radio"
    )

    return view_choice if view_choice != current_view else None


def render_results_header(total_results: int, search_params: Dict[str, Any]) -> None:
    """Отрисовывает заголовок результатов поиска"""
    if total_results > 0:
        query_text = search_params.get("query", "")
        mass_text = ""
        if search_params.get("mass"):
            mass_text = f" по массе {search_params['mass']} Da"

        st.success(f"✅ Найдено {total_results} результатов" +
                  (f" по запросу '{query_text}'" if query_text else "") +
                  mass_text)
    else:
        st.warning("🔍 Результаты не найдены. Попробуйте изменить параметры поиска.")


def render_close_details_buttons() -> Dict[str, bool]:
    """Отрисовывает кнопки закрытия деталей"""
    buttons_pressed = {}

    # Кнопка закрытия деталей метаболита
    if st.session_state.get("show_metabolite_details"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buttons_pressed["close_metabolite"] = st.button(
                "❌ Закрыть детали",
                key="close_met_details",
                use_container_width=True
            )

    # Кнопка закрытия деталей фермента
    if st.session_state.get("show_enzyme_details"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buttons_pressed["close_enzyme"] = st.button(
                "❌ Закрыть детали",
                key="close_enz_details",
                use_container_width=True
            )

    # Кнопка закрытия деталей белка
    if st.session_state.get("show_protein_details"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buttons_pressed["close_protein"] = st.button(
                "❌ Закрыть детали",
                key="close_prot_details",
                use_container_width=True
            )

    # Кнопка закрытия деталей углевода
    if st.session_state.get("show_carbohydrate_details"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buttons_pressed["close_carbohydrate"] = st.button(
                "❌ Закрыть детали",
                key="close_carb_details",
                use_container_width=True
            )

    # Кнопка закрытия деталей липида
    if st.session_state.get("show_lipid_details"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            buttons_pressed["close_lipid"] = st.button(
                "❌ Закрыть детали",
                key="close_lip_details",
                use_container_width=True
            )

    return buttons_pressed
