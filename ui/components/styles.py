"""
CSS-стили для приложения
"""

BASE_STYLES = """
<style>
/* ====== Улучшенная цветовая схема с консистентными градиентами ====== */
:root {
    --primary: #273343;
    --primary-500: #7C3AED;
    --primary-600: #6D28D9;
    --primary-700: #5B21B6;
    --primary-800: #4C1D95;
    --accent: #10B981;
    --accent-light: #34D399;
    --accent-dark: #059669;
    --warning: #F59E0B;
    --warning-light: #FBBF24;
    --danger: #EF4444;
    --danger-light: #F87171;
    --success: #22C55E;
    --info: #3B82F6;
    --bg-start: #0f1220;
    --bg-end: #1b1d2a;
    --bg-card: rgba(255,255,255,0.04);
    --text: #FAFAFA;
    --text-secondary: #E5E7EB;
    --muted: #9CA3AF;
    --muted-light: #D1D5DB;
    --glass: rgba(255,255,255,0.06);
    --glass-strong: rgba(255,255,255,0.10);
    --glass-ultra: rgba(255,255,255,0.15);
    --border: rgba(255,255,255,0.12);
    --border-light: rgba(255,255,255,0.08);
    --shadow-1: 0 10px 30px rgba(0,0,0,0.25);
    --shadow-2: 0 16px 40px rgba(0,0,0,0.35);
    --shadow-3: 0 25px 60px rgba(0,0,0,0.45);
    
    /* Градиенты для разных типов */
    --gradient-primary: linear-gradient(135deg, #273343, #273343);
    --gradient-accent: linear-gradient(135deg, var(--accent), var(--accent-dark));
    --gradient-card: linear-gradient(135deg, var(--glass-strong), var(--glass));
    --gradient-hover: linear-gradient(135deg, var(--glass-ultra), var(--glass-strong));
}

/* Фоновая заливка приложения */
.stApp {
    background: radial-gradient(1200px 600px at 10% 10%, rgba(124, 58, 237, 0.12), transparent),
                radial-gradient(1000px 500px at 90% 20%, rgba(16, 185, 129, 0.10), transparent),
                linear-gradient(180deg, var(--bg-start), var(--bg-end));
}

/* Улучшение expander, контейнеров и блоков */
.block-container { padding-top: 2rem; }

/* ====== Улучшенные карточки результатов ====== */
.card {
    background: var(--gradient-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 12px;
    border: 1px solid var(--border-light);
    box-shadow: var(--shadow-1);
    padding: 1.125rem 1.125rem 1rem 1.125rem;
    margin-bottom: 0.875rem;
    min-height: 180px;
    height: auto;
    display: flex;
    flex-direction: column;
    transition: transform .3s cubic-bezier(0.4, 0, 0.2, 1), 
                box-shadow .3s cubic-bezier(0.4, 0, 0.2, 1), 
                border-color .3s ease,
                background .3s ease;
}

.card:hover {
    box-shadow: var(--shadow-2);
    border-color: var(--primary-500);
    background: var(--gradient-hover);
}

.card-title {
    font-size: 18px;
    font-weight: 800;
    margin: 0 0 8px 0;
    color: var(--text);
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
    line-height: 1.3;
    height: auto;
    overflow: visible;
}

.card-subtitle {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 12px;
    word-wrap: break-word;
    overflow-wrap: break-word;
    line-height: 1.5;
    white-space: pre-line;
    flex-grow: 1;
    height: auto;
    overflow: visible;
    text-overflow: ellipsis;
    display: block;
}

.row-divider { height: 8px; }

/* Ссылки внутри карточек */
.ext-link a {
    text-decoration: none;
    font-size: 14px;
    color: #fff;
    transition: color .2s ease, opacity .2s ease;
    display: inline-block;
    margin: 2px 0;
}
.ext-link a:hover { text-decoration: underline; color: var(--primary-600); }

/* ====== Плашки (pills) ====== */
.pill {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(124,58,237,0.08));
    color: #E5E7EB;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid rgba(124,58,237,0.3);
    margin-right: 6px;
    margin-bottom: 6px;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    text-transform: uppercase;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
}

.pill:hover {
    background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(124,58,237,0.15));
    box-shadow: 0 2px 6px rgba(124,58,237,0.2);
}

/* ====== Наведение на кликабельную карточку ====== */
.clickable-card { position: relative; }
.clickable-card:hover { border-color: rgba(124,58,237,0.45); }

.card-hint {
    color: var(--primary-500);
    font-size: 12px;
    font-style: italic;
    margin-top: 8px;
    text-align: center;
    opacity: 0.85;
    transition: color .2s ease, opacity .2s ease;
}

.clickable-card:hover .card-hint { opacity: 1; color: var(--primary-600); }

.card-hint a {
    color: var(--primary-500) !important;
    text-decoration: none !important;
    font-weight: 600 !important;
}
.card-hint a:hover { color: var(--primary-600) !important; text-decoration: underline !important; }

/* ====== Формулы ====== */
.formula { font-family: 'Times New Roman', serif; font-style: normal; }
.formula .subscript { font-size: 0.7em; vertical-align: sub; font-weight: normal; }
.formula .superscript { font-size: 0.7em; vertical-align: super; font-weight: normal; }

/* ====== Статистика ====== */
.stats-card {
    text-align: center;
    padding: 0.75rem;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 12px;
    min-width: 88px;
}
.stats-title { font-size: 0.85rem; color: var(--muted); font-weight: 600; }
.stats-value { font-size: 1.25rem; font-weight: 700; color: var(--text); }

/* ====== Формы / Инпуты / Кнопки ====== */
.form-container {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border-radius: 20px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}
.form-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text); }

/* Улучшенные кнопки Streamlit */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.25rem !important;
    border: 1px solid var(--border) !important;
    color: white !important;
    background: var(--gradient-primary) !important;
    box-shadow: var(--shadow-1) !important;
    transition: transform .2s cubic-bezier(0.4, 0, 0.2, 1), 
                box-shadow .2s cubic-bezier(0.4, 0, 0.2, 1), 
                filter .2s ease !important;
    letter-spacing: 0.025em !important;
    scroll-behavior: auto !important;
}

/* Предотвращение автоматической прокрутки при нажатии на кнопки */
.stButton > button:focus {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки для всех интерактивных элементов */
.stButton > button,
.stButton > button:active,
.stButton > button:focus,
.stButton > button:hover {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
    scroll-padding: 0 !important;
}

.stButton > button:hover {
    box-shadow: var(--shadow-2) !important;
    filter: brightness(1.05) !important;
    border-color: var(--border-light) !important;
}

.stButton > button:active { 
    transform: translateY(0) scale(1) !important; 
    box-shadow: var(--shadow-1) !important;
}

/* Дополнительные стили для предотвращения прокрутки */
.stButton {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки для основного контейнера */
.main .block-container {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки для всех интерактивных элементов */
[data-testid="stButton"],
[data-testid="stButton"] > button,
.stButton,
.stButton > button {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
    scroll-padding: 0 !important;
}

/* Предотвращение прокрутки для форм и инпутов */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiselect > div > div > div {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки на уровне HTML и body */
html, body {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
    scroll-padding: 0 !important;
}

/* Предотвращение прокрутки для Streamlit элементов */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="main"] {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Дополнительные стили для предотвращения прокрутки */
.stButton > button:focus-visible {
    outline: none !important;
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки для всех форм */
.stForm {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки для результатов поиска */
.results-container {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки при навигации клавиатурой */
.stButton > button:focus-within {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Предотвращение прокрутки для всех интерактивных элементов */
button:focus,
input:focus,
select:focus,
textarea:focus {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
    scroll-padding: 0 !important;
}

/* Предотвращение прокрутки для модальных окон */
.stModal {
    scroll-behavior: auto !important;
    scroll-margin: 0 !important;
}

/* Кнопки в модальных окнах - одинаковый размер */
.stButton > button[key*="copy_"],
.stButton > button[key*="refresh_"],
.stButton > button[key*="close_"] {
    width: 100% !important;
    min-width: 120px !important;
    max-width: 150px !important;
    height: 40px !important;
    font-size: 14px !important;
    border-radius: 8px !important;
}

/* Контейнеры колонок для кнопок */
[data-testid="column"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* ====== Результаты ====== */
.results-container { margin-top: 1rem; }
.results-header { font-size: 1.5rem; font-weight: 700; margin-bottom: 1rem; color: var(--text); }
.results-count { color: var(--accent); font-weight: 600; }

/* ====== Пагинация ====== */
.pagination-container {
    display: flex; justify-content: center; align-items: center;
    gap: 1rem; margin: 2rem 0;
}
.pagination-button {
    padding: 0.5rem 1rem; border-radius: 4px; border: 1px solid var(--primary-600);
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(124,58,237,0.04));
    color: #EDE9FE; cursor: pointer; transition: all 0.2s ease;
}
.pagination-button:hover:not(:disabled) { background: var(--primary-600); color: white; }
.pagination-button:disabled { opacity: 0.55; cursor: not-allowed; }
.pagination-info { color: var(--muted); font-size: 0.95rem; }

/* ====== Детали ====== */
.details-container {
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
    border-radius: 20px; padding: 1.5rem; margin: 1rem 0;
    border: 1px solid var(--border);
}
.details-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 1rem; color: var(--text); }
.details-section { margin-bottom: 1.5rem; }
.details-section-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text); }
.details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.details-item { background: var(--glass); padding: 0.9rem; border-radius: 12px; border: 1px solid var(--border); }
.details-label { font-size: 0.9rem; color: var(--muted); margin-bottom: 0.25rem; }
.details-value { font-size: 1rem; color: var(--text); font-weight: 600; }

/* ====== Внешние ссылки ====== */
.external-links { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.6rem; margin-top: 1rem; }
.external-link { padding: 0.7rem; background: var(--glass); border: 1px solid var(--border); border-radius: 12px; text-align: center; }
.external-link a { color: var(--primary-500) !important; text-decoration: none !important; font-weight: 600 !important; }
.external-link a:hover { text-decoration: underline !important; color: var(--primary-600) !important; }

/* ====== Метрики ====== */
.metric-card { background: var(--glass); padding: 1rem; border-radius: 12px; text-align: center; border: 1px solid var(--border); }
.metric-value { font-size: 1.5rem; font-weight: 700; color: var(--text); }
.metric-label { font-size: 0.9rem; color: var(--muted); }

/* ====== 3D Визуализация в модальных окнах ====== */
.structure-card {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem;
    margin: 1rem 0;
    overflow: hidden;
    max-width: 100%;
    box-sizing: border-box;
}

/* ====== 3D и 2D Структуры ====== */
.structure-3d-container {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem auto;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    overflow: hidden;
    position: relative;
    min-height: 400px;
    display: flex;
    align-items: center;
    justify-content: center;
    max-width: 100%;
    text-align: center;
}

.structure-2d-container {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem auto;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    max-width: 100%;
    text-align: center;
}

.structure-2d-image {
    background: white;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    margin: 1rem auto;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 450px;
    max-width: 100%;
}

.structure-2d-properties {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    margin: 1rem 0;
}

.structure-2d-properties p {
    margin: 0.5rem 0;
    line-height: 1.6;
}

.structure-2d-metric {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(124,58,237,0.06));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 8px;
    padding: 0.8rem;
    margin: 0.5rem 0;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.structure-2d-metric:hover {
    box-shadow: 0 4px 12px rgba(124,58,237,0.2);
}

.structure-2d-metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--primary-500);
    margin-bottom: 0.2rem;
    text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.structure-2d-metric-label {
    font-size: 0.85rem;
    color: var(--muted);
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ====== Адаптивность для различных размеров экрана ====== */
@media (max-width: 1200px) {
    .structure-3d-container {
        min-height: 350px;
        padding: 0.75rem;
    }
    
    .stApp [data-testid="stHorizontalBlock"] iframe {
        max-width: 100% !important;
        height: auto !important;
    }
    
    .card {
        height: auto;
        min-height: 180px;
        padding: 16px;
    }

    .card-title {
        font-size: 17px;
        line-height: 1.2;
        -webkit-line-clamp: 2;
        max-height: 3em;
    }
    
    .card-subtitle {
        -webkit-line-clamp: 3;
        font-size: 13px;
    }
}

/* Стили для iframe и встроенного контента */
.stApp iframe {
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    max-width: 100% !important;
    width: 100% !important;
    overflow: hidden !important;
    background: rgba(255,255,255,0.02);
}

/* Улучшенная поддержка темной темы */
@media (prefers-color-scheme: dark) {
    .structure-2d-image {
        background: rgba(255,255,255,0.98);
        border: 1px solid rgba(0,0,0,0.1);
    }
    
    .structure-3d-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    }
}

/* Контейнеры для визуализации */
[data-testid="stHorizontalBlock"] {
    overflow: hidden !important;
    border-radius: 12px !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* Улучшения для радио-кнопок */
.stRadio > div {
    gap: 0.75rem;
    align-items: center;
}

.stRadio > div > label {
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.5rem 1rem;
    transition: all 0.2s ease;
    cursor: pointer;
}

.stRadio > div > label:hover {
    background: rgba(124,58,237,0.1);
    border-color: var(--primary-500);
}

/* Улучшения для информационных блоков */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
}

/* ====== Мобильная адаптивность ====== */
@media (max-width: 768px) {
    .block-container { padding-top: 1rem; }
    
    .details-grid { grid-template-columns: 1fr; }
    .external-links { grid-template-columns: 1fr; }
    
    /* Карточки на мобильных */
    .card { 
        padding: 16px; 
        height: auto; 
        min-height: 160px;
    }
    
    .card-title { 
        font-size: 17px;
        line-height: 1.2;
        height: auto;
        overflow: visible;
    }
    
    .card-subtitle {
        height: auto;
        overflow: visible;
        font-size: 13px;
    }
    
    /* 3D визуализация на мобильных */
    .structure-3d-container {
        min-height: 280px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .structure-card {
        padding: 0.75rem;
        margin: 0.75rem 0;
    }
    
    .structure-2d-container {
        padding: 1rem;
        margin: 0.75rem 0;
    }
    
    .structure-2d-image {
        min-height: 280px;
        padding: 1rem;
    }
    
    /* Формы на мобильных */
    .form-container {
        padding: 0.75rem;
        border-radius: 12px;
    }
    
    /* Кнопки на мобильных */
    .stButton > button {
        padding: 0.5rem 0.75rem !important;
        font-size: 13px !important;
        border-radius: 6px !important;
    }
    
    /* Статистика на мобильных */
    .stats-card {
        padding: 0.5rem;
        min-width: 70px;
        border-radius: 12px;
    }
    
    .stats-title { font-size: 0.8rem; }
    .stats-value { font-size: 1.1rem; }
}
</style>
"""


def inject_styles() -> None:
    """Внедряет базовые CSS-стили в Streamlit"""
    import streamlit as st
    st.markdown(BASE_STYLES, unsafe_allow_html=True)
