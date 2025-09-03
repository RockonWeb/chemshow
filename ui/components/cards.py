"""
Компоненты карточек для отображения результатов поиска
"""
import streamlit as st
from typing import Dict, Any, Optional
from .utils import (
    get_display_name, safe_get_value, format_mass, format_chemical_formula,
    create_external_links, create_pills_list, truncate_description
)
from .comparison import add_to_comparison_button, comparison_comparator

# Импорт функции для безопасного открытия диалогов
try:
    from ..main import open_dialog_safely
except ImportError:
    # Fallback для случаев, когда main.py не доступен
    def open_dialog_safely(dialog_type: str, entity: Dict[str, Any]):
        """Резервная функция открытия диалога"""
        # Закрываем все диалоги
        st.session_state.show_metabolite_details = False
        st.session_state.show_enzyme_details = False
        st.session_state.show_protein_details = False
        st.session_state.show_carbohydrate_details = False
        st.session_state.show_lipid_details = False

        # Очищаем выбранные элементы
        st.session_state.selected_metabolite = None
        st.session_state.selected_enzyme = None
        st.session_state.selected_protein = None
        st.session_state.selected_carbohydrate = None
        st.session_state.selected_lipid = None

        # Открываем нужный диалог
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


def render_metabolite_card(metabolite: Dict[str, Any], card_key: str) -> None:
    """Карточка метаболита с ссылками и кнопкой деталей"""
    # Получаем данные
    name = get_display_name(metabolite)
    formula = safe_get_value(metabolite, "formula", "—")
    mass = safe_get_value(metabolite, "exact_mass")
    mass_str = format_mass(mass, "Da")
    class_name = safe_get_value(metabolite, "class_name", "—")

    # Создаем ссылки
    links = create_external_links("metabolite", metabolite)
    links_html = ""
    if links:
        link_items = [f"<span class='ext-link'><a href='{link['url']}' target='_blank'>{link['name']}</a></span>" for link in links]
        links_html = " &middot; ".join(link_items)

    # Создаем pills для класса
    pills_html = create_pills_list([class_name]) if class_name != "—" else ""

    # Форматируем формулу
    formatted_formula = format_chemical_formula(formula)

    # Создаем карточку
    st.markdown(
        f"""
        <div class="card clickable-card" style="cursor: pointer;">
          <div class="card-title">{name}</div>
          <div class="card-subtitle">Формула: <b>{formatted_formula}</b><br>Масса: <b>{mass_str}</b></div>
          <div>{pills_html}</div>
          <div class="row-divider"></div>
          <div>{links_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Кнопки действий
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Показать детали", key=card_key, width='stretch'):
            # Используем безопасную функцию открытия диалога
            open_dialog_safely("metabolite", metabolite)

    with col2:
        add_to_comparison_button(metabolite, "metabolites", comparison_comparator)


def render_enzyme_card(enzyme: Dict[str, Any], card_key: str) -> None:
    """Карточка фермента с ссылками и кнопкой деталей"""
    # Получаем данные
    name = get_display_name(enzyme)
    ec_number = safe_get_value(enzyme, "ec_number", "—")
    organism = safe_get_value(enzyme, "organism", "—")
    family = safe_get_value(enzyme, "family", "—")

    # Создаем ссылки
    links = create_external_links("enzyme", enzyme)
    links_html = ""
    if links:
        link_items = [f"<span class='ext-link'><a href='{link['url']}' target='_blank'>{link['name']}</a></span>" for link in links]
        links_html = " &middot; ".join(link_items)

    # Создаем subtitle
    props = []
    if ec_number != "—":
        props.append(f"EC: <b>{ec_number}</b>")
    if organism != "—":
        props.append(f"Организм: <b>{organism}</b>")
    if family != "—":
        props.append(f"Семейство: <b>{family}</b>")
    subtitle = "<br>".join(props)

    # Создаем карточку
    st.markdown(
        f"""
        <div class="card clickable-card" style="cursor: pointer;">
          <div class="card-title">{name}</div>
          <div class="card-subtitle">{subtitle}</div>
          <div class="row-divider"></div>
          <div>{links_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Кнопки действий
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Показать детали", key=card_key, width='stretch'):
            # Используем безопасную функцию открытия диалога
            open_dialog_safely("enzyme", enzyme)

    with col2:
        add_to_comparison_button(enzyme, "enzymes", comparison_comparator)


def render_protein_card(protein: Dict[str, Any], card_key: str) -> None:
    """Карточка белка с ссылками и кнопкой деталей"""
    # Получаем данные
    name = get_display_name(protein)
    function = safe_get_value(protein, "function", "—")
    organism = safe_get_value(protein, "organism", "—")
    family = safe_get_value(protein, "family", "—")

    # Создаем ссылки
    links = create_external_links("protein", protein)
    links_html = ""
    if links:
        link_items = [f"<span class='ext-link'><a href='{link['url']}' target='_blank'>{link['name']}</a></span>" for link in links]
        links_html = " &middot; ".join(link_items)

    # Создаем subtitle
    props = []
    if function != "—":
        truncated_func = truncate_description(function)
        props.append(f"Функция: <b>{truncated_func}</b>")
    if organism != "—":
        props.append(f"Организм: <b>{organism}</b>")
    if family != "—":
        props.append(f"Семейство: <b>{family}</b>")
    subtitle = "<br>".join(props)

    # Создаем карточку
    st.markdown(
        f"""
        <div class="card clickable-card" style="cursor: pointer;">
          <div class="card-title">{name}</div>
          <div class="card-subtitle">{subtitle}</div>
          <div class="row-divider"></div>
          <div>{links_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Кнопки действий
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Показать детали", key=card_key, width='stretch'):
            # Используем безопасную функцию открытия диалога
            open_dialog_safely("protein", protein)

    with col2:
        add_to_comparison_button(protein, "proteins", comparison_comparator)


def render_carbohydrate_card(carbohydrate: Dict[str, Any], card_key: str) -> None:
    """Карточка углевода с ссылками и кнопкой деталей"""
    # Получаем данные
    name = get_display_name(carbohydrate)
    formula = safe_get_value(carbohydrate, "formula", "—")
    mass = safe_get_value(carbohydrate, "exact_mass")
    mass_str = format_mass(mass, "Da")
    carb_type = safe_get_value(carbohydrate, "type", "—")

    # Создаем pills для типа
    pills_html = create_pills_list([carb_type]) if carb_type != "—" else ""

    # Форматируем формулу
    formatted_formula = format_chemical_formula(formula)

    # Создаем карточку
    st.markdown(
        f"""
        <div class="card clickable-card" style="cursor: pointer;">
          <div class="card-title">{name}</div>
          <div class="card-subtitle">Формула: <b>{formatted_formula}</b><br>Масса: <b>{mass_str}</b></div>
          <div>{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Кнопки действий
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Показать детали", key=card_key, width='stretch'):
            # Используем безопасную функцию открытия диалога
            open_dialog_safely("carbohydrate", carbohydrate)

    with col2:
        add_to_comparison_button(carbohydrate, "carbohydrates", comparison_comparator)


def render_lipid_card(lipid: Dict[str, Any], card_key: str) -> None:
    """Карточка липида с ссылками и кнопкой деталей"""
    # Получаем данные
    name = get_display_name(lipid)
    formula = safe_get_value(lipid, "formula", "—")
    mass = safe_get_value(lipid, "exact_mass")
    mass_str = format_mass(mass, "Da")
    lipid_type = safe_get_value(lipid, "type", "—")

    # Создаем pills для типа
    pills_html = create_pills_list([lipid_type]) if lipid_type != "—" else ""

    # Форматируем формулу
    formatted_formula = format_chemical_formula(formula)

    # Создаем карточку
    st.markdown(
        f"""
        <div class="card clickable-card" style="cursor: pointer;">
          <div class="card-title">{name}</div>
          <div class="card-subtitle">Формула: <b>{formatted_formula}</b><br>Масса: <b>{mass_str}</b></div>
          <div>{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Кнопки действий
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Показать детали", key=card_key, width='stretch'):
            # Используем безопасную функцию открытия диалога
            open_dialog_safely("lipid", lipid)

    with col2:
        add_to_comparison_button(lipid, "lipids", comparison_comparator)
