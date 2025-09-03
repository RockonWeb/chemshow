"""
Компоненты для отображения детальной информации о сущностях
"""
import streamlit as st
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Добавляем путь к config для корректных импортов
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
# Импорт вспомогательных функций
try:
    # Try absolute import first
    from components.utils import (
        get_display_name, safe_get_value, format_mass, format_chemical_formula,
        create_external_links, create_metric_html
    )
except ImportError:
    # Fallback to relative import
    try:
        from .utils import (
            get_display_name, safe_get_value, format_mass, format_chemical_formula,
            create_external_links, create_metric_html
        )
    except ImportError:
        from utils import (
            get_display_name, safe_get_value, format_mass, format_chemical_formula,
            create_external_links, create_metric_html
        )

# Улучшенные адаптивные стили для модальных окон
modal_styles = """
<style>
/* Адаптивные стили для модальных окон */
.stDialog > div > div > div {
    border-radius: 16px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.97) 0%, rgba(250, 250, 255, 0.97) 100%) !important;
    backdrop-filter: blur(15px) !important;
    max-width: min(90vw, 900px) !important;
    max-height: 85vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

.stDialog > div > div > div > div > div {
    background: transparent !important;
    padding: 1rem !important;
}

/* Улучшенные стили для кнопок */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    letter-spacing: 0.3px !important;
    min-height: 38px !important;
}

.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.18) !important;
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Адаптивные цветовые схемы */
.metabolite-theme {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: -1rem -1rem 1.5rem -1rem !important;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25) !important;
    position: relative;
    overflow: hidden;
}

.metabolite-theme::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    z-index: -1;
}

.enzyme-theme {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin: -1rem -1rem 1.5rem -1rem !important;
    box-shadow: 0 8px 30px rgba(240, 147, 251, 0.25) !important;
    position: relative;
    overflow: hidden;
}

.enzyme-theme::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    z-index: -1;
}

/* Адаптивные карточки с градиентными фонами */
.info-card {
    background: linear-gradient(135deg, #e3f2fd 0%, #f0f4ff 100%) !important;
    padding: 1.2rem !important;
    border-radius: 10px !important;
    border-left: 3px solid #2196f3 !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 3px 12px rgba(33, 150, 243, 0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.info-card:hover {
    box-shadow: 0 6px 20px rgba(33, 150, 243, 0.15) !important;
}

.properties-card {
    background: linear-gradient(135deg, #f3e5f5 0%, #faf4ff 100%) !important;
    padding: 1.2rem !important;
    border-radius: 10px !important;
    border-left: 3px solid #9c27b0 !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 3px 12px rgba(156, 39, 176, 0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.properties-card:hover {
    box-shadow: 0 6px 20px rgba(156, 39, 176, 0.15) !important;
}

.structure-card {
    background: linear-gradient(135deg, #e8f5e8 0%, #f4fdf4 100%) !important;
    padding: 1.2rem !important;
    border-radius: 10px !important;
    text-align: center !important;
    border: 2px dashed rgba(76, 175, 80, 0.4) !important;
    margin-bottom: 1rem !important;
    transition: border-color 0.2s ease !important;
}

.structure-card:hover {
    border-color: rgba(76, 175, 80, 0.7) !important;
}

.description-card {
    background: linear-gradient(135deg, #fce4ec 0%, #fef7f7 100%) !important;
    padding: 1.2rem !important;
    border-radius: 10px !important;
    border-left: 3px solid #e91e63 !important;
    box-shadow: 0 3px 12px rgba(233, 30, 99, 0.08) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.description-card:hover {
    box-shadow: 0 6px 20px rgba(233, 30, 99, 0.15) !important;
}

/* Мобильная адаптация модальных окон */
@media (max-width: 768px) {
    .stDialog > div > div > div {
        max-width: 95vw !important;
        max-height: 90vh !important;
        border-radius: 12px !important;
        margin: 1rem !important;
    }
    
    .metabolite-theme, .enzyme-theme {
        padding: 1rem !important;
        margin: -0.5rem -0.5rem 1rem -0.5rem !important;
        border-radius: 8px !important;
    }
    
    .info-card, .properties-card, .description-card {
        padding: 1rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.75rem !important;
    }
    
    .structure-card {
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    .stButton > button {
        font-size: 12px !important;
        padding: 8px 16px !important;
        min-height: 36px !important;
    }
}
</style>
"""

st.markdown(modal_styles, unsafe_allow_html=True)


@st.dialog("🧬 Детали метаболита")
def show_metabolite_details(metabolite: Dict[str, Any]) -> None:
    """Показывает детальную информацию о метаболите в красивом модальном окне с 3D визуализацией"""

    # Получаем данные метаболита
    display_name = get_display_name(metabolite, "Метаболит")
    formula = format_chemical_formula(safe_get_value(metabolite, "formula", "Не указано"))
    mass = safe_get_value(metabolite, 'exact_mass')
    mass_str = format_mass(mass, "Da")

    # Красивый заголовок с градиентным фоном
    st.markdown('<div class="metabolite-theme">', unsafe_allow_html=True)
    st.markdown(f"# 🧬 {display_name}")
    st.markdown(f"**⚗️ Формула:** {formula} | **⚖️ Масса:** {mass_str}", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Основная информация в красивых карточках
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Основная информация")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown(f"**🇺🇸 Название (EN):** {safe_get_value(metabolite, 'name', 'Не указано')}")
        st.markdown(f"**🇷🇺 Название (RU):** {safe_get_value(metabolite, 'name_ru', 'Не указано')}")
        st.markdown(f"**🏷️ Класс:** {safe_get_value(metabolite, 'class_name', 'Не указано')}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚖️ Физико-химические свойства")
        st.markdown('<div class="properties-card">', unsafe_allow_html=True)
        st.markdown(f"**⚗️ Химическая формула:** {formula}", unsafe_allow_html=True)
        st.markdown(f"**🏋️ Молекулярная масса:** {mass_str}")

        # Внешние ссылки
        hmdb_id = safe_get_value(metabolite, 'hmdb_id')
        kegg_id = safe_get_value(metabolite, 'kegg_id')
        chebi_id = safe_get_value(metabolite, 'chebi_id')
        pubchem_id = safe_get_value(metabolite, 'pubchem_cid')

        if any([hmdb_id, kegg_id, chebi_id, pubchem_id]):
            st.markdown("### 🔗 Внешние ссылки")
            links = []
            if hmdb_id:
                links.append(f"🔗 [HMDB]({EXTERNAL_LINKS['hmdb_base']}{hmdb_id})")
            if kegg_id:
                links.append(f"🔗 [KEGG]({EXTERNAL_LINKS['kegg_base']}{kegg_id})")
            if chebi_id:
                links.append(f"🔗 [ChEBI]({EXTERNAL_LINKS['chebi_base']}{chebi_id})")
            if pubchem_id:
                links.append(f"🔗 [PubChem]({EXTERNAL_LINKS['pubchem_base']}{pubchem_id})")
            st.markdown(" | ".join(links))

        st.markdown('</div>', unsafe_allow_html=True)

    # 3D визуализация в модальном окне с таба
    smiles = safe_get_value(metabolite, 'smiles') or safe_get_value(metabolite, 'smiles_string')
    if smiles and smiles != 'Не указано':
        st.markdown("### 🧬 Визуализация молекулярной структуры")

        # Создаем табы для разных видов визуализации
        tab1, tab2 = st.tabs(["🧬 3D Структура", "🖼️ 2D Структура"])

        with tab1:

            # Показываем 3D структуру в модальном окне с оптимальными размерами
            try:
                from .visualization_3d import render_3d_structure
                # Компактные размеры для модального окна (адаптивно)
                modal_width = min(400, 350)  # Оптимально для модальных окон
                modal_height = min(300, 250)  # Компактное отображение
                render_3d_structure(smiles, f"3D структура: {display_name}", 
                                  width=modal_width, height=modal_height)
            except Exception as e:
                st.error(f"❌ Ошибка загрузки 3D структуры: {str(e)[:100]}...") 
                st.info("💡 Попробуйте 2D визуализацию или проверьте SMILES строку")
                st.code(smiles, language="text")



        with tab2:

            st.markdown("*Детальный анализ молекулярных свойств*")

            # Показываем 2D структуру и свойства
            try:
                from .visualization_3d import render_2d_structure
                render_2d_structure(smiles, f"2D структура: {display_name}")
            except Exception as e:
                st.error(f"❌ Ошибка загрузки 2D структуры: {str(e)[:100]}...")
                st.info("💡 Проверьте SMILES строку")
                st.code(smiles, language="text")

    else:
        st.info("💡 Для этого метаболита нет доступных данных о структуре (SMILES)")

    # Описание с красивым оформлением
    description = safe_get_value(metabolite, 'description')
    if description and description != 'Не указано':
        st.markdown("### 📝 Описание и биологическая роль")
        st.markdown('<div class="description-card">', unsafe_allow_html=True)
        st.info(description)
        st.markdown('</div>', unsafe_allow_html=True)

    # Кнопки управления с красивым дизайном
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Копировать", key="copy_metabolite_modal", width='stretch'):
            # Имитируем копирование данных
            data_to_copy = f"""
Название: {display_name}
Формула: {formula}
Масса: {mass_str}
Класс: {safe_get_value(metabolite, 'class_name', 'Не указано')}
"""
            st.session_state.copied_data = data_to_copy
            st.success("✅ Данные скопированы!")

    with col2:
        if st.button("Обновить", key="refresh_metabolite_modal", width='stretch'):
            st.rerun()

    with col3:
        if st.button("❌ Закрыть", key="close_metabolite_modal", type="primary"):
            # Закрываем все диалоги
            st.session_state.show_metabolite_details = False
            st.session_state.show_enzyme_details = False
            st.session_state.show_protein_details = False
            st.session_state.show_carbohydrate_details = False
            st.session_state.show_lipid_details = False


@st.dialog("🧬 Детали фермента")
def show_enzyme_details(enzyme: Dict[str, Any]) -> None:
    """Показывает детальную информацию о ферменте в красивом модальном окне"""

    # Получаем данные фермента
    display_name = get_display_name(enzyme, "Фермент")
    ec_number = safe_get_value(enzyme, 'ec_number', 'Не указано')
    systematic_name = safe_get_value(enzyme, 'systematic_name', 'Не указано')

    # Красивый заголовок с градиентным фоном для ферментов
    st.markdown('<div class="enzyme-theme">', unsafe_allow_html=True)
    st.markdown(f"# 🧬 {display_name}")
    st.markdown(f"**🔢 EC номер:** {ec_number} | **⚗️ Систематическое название:** {systematic_name[:50]}...")
    st.markdown('</div>', unsafe_allow_html=True)

    # Основная информация в красивых карточках
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Основная информация")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown(f"**🇺🇸 Название (EN):** {safe_get_value(enzyme, 'name', 'Не указано')}")
        st.markdown(f"**🇷🇺 Название (RU):** {safe_get_value(enzyme, 'name_ru', 'Не указано')}")
        st.markdown(f"**🔢 EC номер:** {ec_number}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🧪 Функциональные характеристики")
        st.markdown('<div class="properties-card">', unsafe_allow_html=True)
        st.markdown(f"**🏷️ Класс:** {safe_get_value(enzyme, 'class_name', 'Не указано')}")
        st.markdown(f"**⚡ Тип реакции:** {safe_get_value(enzyme, 'reaction_type', 'Не указано')}")

        # Внешние ссылки
        brenda_id = safe_get_value(enzyme, 'brenda_id')
        kegg_id = safe_get_value(enzyme, 'kegg_id')
        uniprot_id = safe_get_value(enzyme, 'uniprot_id')

        if any([brenda_id, kegg_id, uniprot_id]):
            st.markdown("### 🔗 Внешние ссылки")
            links = []
            if brenda_id:
                links.append(f"🔗 [BRENDA](https://www.brenda-enzymes.org/enzyme.php?ecno={brenda_id})")
            if kegg_id:
                links.append(f"🔗 [KEGG]({EXTERNAL_LINKS['kegg_base']}{kegg_id})")
            if uniprot_id:
                links.append(f"🔗 [UniProt](https://www.uniprot.org/uniprotkb/{uniprot_id})")
            st.markdown(" | ".join(links))

        st.markdown('</div>', unsafe_allow_html=True)

    # Описание
    description = safe_get_value(enzyme, 'description')
    if description and description != 'Не указано':
        st.markdown("### 📝 Описание и механизм действия")
        st.markdown('<div class="description-card">', unsafe_allow_html=True)
        st.info(description)
        st.markdown('</div>', unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Копировать", key="copy_enzyme_modal", width='stretch'):
            data_to_copy = f"""
Название: {display_name}
EC номер: {ec_number}
Класс: {safe_get_value(enzyme, 'class_name', 'Не указано')}
"""
            st.session_state.copied_data = data_to_copy
            st.success("✅ Данные скопированы!")

    with col2:
        if st.button("Обновить", key="refresh_enzyme_modal", width='stretch'):
            st.rerun()

    with col3:
        if st.button("❌ Закрыть", key="close_enzyme_modal", type="primary"):
            # Закрываем все диалоги
            st.session_state.show_metabolite_details = False
            st.session_state.show_enzyme_details = False
            st.session_state.show_protein_details = False
            st.session_state.show_carbohydrate_details = False
            st.session_state.show_lipid_details = False


@st.dialog("🧬 Детали белка")
def show_protein_details(protein: Dict[str, Any]) -> None:
    """Показывает детальную информацию о белке в модальном окне с использованием st.dialog()"""

    # Получаем данные белка
    display_name = get_display_name(protein, "Белок")
    sequence = safe_get_value(protein, 'sequence', '')
    length = len(sequence) if sequence else 0

    # Заголовок модального окна
    st.markdown(f"## 🧬 {display_name}")

    # Основная информация
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Основная информация")
        with st.container():
            st.info(f"""
            **Название (EN):** {safe_get_value(protein, 'name', 'Не указано')}

            **Название (RU):** {safe_get_value(protein, 'name_ru', 'Не указано')}

            **Длина последовательности:** {length} аминокислот

            **Тип:** {safe_get_value(protein, 'protein_type', 'Не указано')}
            """)

    with col2:
        st.markdown("### 🧪 Характеристики")
        with st.container():
            st.success(f"""
            **UniProt ID:** {safe_get_value(protein, 'uniprot_id', 'Не указано')}

            **Функция:** {safe_get_value(protein, 'function', 'Не указано')}
            """)

    # Последовательность (если не слишком длинная)
    if sequence and length <= 500:
        st.markdown("### 🧬 Аминокислотная последовательность")
        st.code(sequence, language="text")

    # Описание
    description = safe_get_value(protein, 'description')
    if description and description != 'Не указано':
        st.markdown("### 📝 Описание")
        st.info(description)


    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Копировать", key="copy_protein_modal", width='stretch'):
            data_to_copy = f"""
Название: {display_name}
Функция: {safe_get_value(protein, 'function', 'Не указано')}
Тип: {safe_get_value(protein, 'protein_type', 'Не указано')}
"""
            st.session_state.copied_data = data_to_copy
            st.success("✅ Данные скопированы!")

    with col2:
        if st.button("Обновить", key="refresh_protein_modal", width='stretch'):
            st.rerun()

    with col3:
        if st.button("❌ Закрыть", key="close_protein_modal", type="primary"):
            # Закрываем все диалоги
            st.session_state.show_metabolite_details = False
            st.session_state.show_enzyme_details = False
            st.session_state.show_protein_details = False
            st.session_state.show_carbohydrate_details = False
            st.session_state.show_lipid_details = False


@st.dialog("🧬 Детали углевода")
def show_carbohydrate_details(carbohydrate: Dict[str, Any]) -> None:
    """Показывает детальную информацию об углеводе в модальном окне с использованием st.dialog()"""

    # Получаем данные углевода
    display_name = get_display_name(carbohydrate, "Углевод")

    # Заголовок модального окна
    st.markdown(f"## 🧬 {display_name}")

    # Основная информация
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Основная информация")
        with st.container():
            st.info(f"""
            **Название (EN):** {safe_get_value(carbohydrate, 'name', 'Не указано')}

            **Название (RU):** {safe_get_value(carbohydrate, 'name_ru', 'Не указано')}

            **Формула:** {safe_get_value(carbohydrate, 'formula', 'Не указано')}

            **Тип:** {safe_get_value(carbohydrate, 'carbohydrate_type', 'Не указано')}
            """)

    with col2:
        st.markdown("### ⚖️ Свойства")
        with st.container():
            st.success(f"""
            **Молекулярная масса:** {safe_get_value(carbohydrate, 'molecular_weight', 'Не указано')}

            **Класс:** {safe_get_value(carbohydrate, 'class_name', 'Не указано')}
            """)

    # Описание
    description = safe_get_value(carbohydrate, 'description')
    if description and description != 'Не указано':
        st.markdown("### 📝 Описание")
        st.info(description)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Копировать", key="copy_carbohydrate_modal", width='stretch'):
            data_to_copy = f"""
Название: {display_name}
Формула: {safe_get_value(carbohydrate, 'formula', 'Не указано')}
Тип: {safe_get_value(carbohydrate, 'carbohydrate_type', 'Не указано')}
"""
            st.session_state.copied_data = data_to_copy
            st.success("✅ Данные скопированы!")

    with col2:
        if st.button("Обновить", key="refresh_carbohydrate_modal", width='stretch'):
            st.rerun()

    with col3:
        if st.button("❌ Закрыть", key="close_carbohydrate_modal", type="primary", width='stretch'):
            # Закрываем все диалоги
            st.session_state.show_metabolite_details = False
            st.session_state.show_enzyme_details = False
            st.session_state.show_protein_details = False
            st.session_state.show_carbohydrate_details = False
            st.session_state.show_lipid_details = False


@st.dialog("🧬 Детали липида")
def show_lipid_details(lipid: Dict[str, Any]) -> None:
    """Показывает детальную информацию о липиде в модальном окне с использованием st.dialog()"""

    # Получаем данные липида
    display_name = get_display_name(lipid, "Липид")

    # Заголовок модального окна
    st.markdown(f"## 🧬 {display_name}")

    # Основная информация
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 Основная информация")
        with st.container():
            st.info(f"""
            **Название (EN):** {safe_get_value(lipid, 'name', 'Не указано')}

            **Название (RU):** {safe_get_value(lipid, 'name_ru', 'Не указано')}

            **Формула:** {safe_get_value(lipid, 'formula', 'Не указано')}

            **Тип:** {safe_get_value(lipid, 'lipid_type', 'Не указано')}
            """)

    with col2:
        st.markdown("### ⚖️ Свойства")
        with st.container():
            st.success(f"""
            **Молекулярная масса:** {safe_get_value(lipid, 'molecular_weight', 'Не указано')}

            **Класс:** {safe_get_value(lipid, 'class_name', 'Не указано')}
            """)

    # Описание
    description = safe_get_value(lipid, 'description')
    if description and description != 'Не указано':
        st.markdown("### 📝 Описание")
        st.info(description)


    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Копировать", key="copy_lipid_modal", width='stretch'):
            data_to_copy = f"""
Название: {display_name}
Формула: {safe_get_value(lipid, 'formula', 'Не указано')}
Тип: {safe_get_value(lipid, 'lipid_type', 'Не указано')}
"""
            st.session_state.copied_data = data_to_copy
            st.success("✅ Данные скопированы!")

    with col2:
        if st.button("Обновить", key="refresh_lipid_modal", width='stretch'):
            st.rerun()

    with col3:
        if st.button("❌ Закрыть", key="close_lipid_modal", type="primary", width='stretch'):
            # Закрываем все диалоги
            st.session_state.show_metabolite_details = False
            st.session_state.show_enzyme_details = False
            st.session_state.show_protein_details = False
            st.session_state.show_carbohydrate_details = False
            st.session_state.show_lipid_details = False
