"""
3D-визуализация молекулярных структур с использованием Py3Dmol и rdkit
"""
import streamlit as st
import streamlit.components.v1 as components
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import hashlib
import os

# Настройка логирования
logger = logging.getLogger(__name__)

# Импорт дополнительных библиотек для продвинутой визуализации
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Попытка импорта rdkit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    # Проверяем, что импорт действительно работает
    test_mol = Chem.MolFromSmiles("C")
    if test_mol is not None:
        RDKIT_AVAILABLE = True
        logger.info("rdkit успешно импортирован и работает")
    else:
        RDKIT_AVAILABLE = False
        logger.warning("rdkit импортирован, но не работает корректно")
        Chem = None
        AllChem = None
except ImportError as e:
    logger.warning(f"rdkit не установлен. 3D-визуализация будет ограничена. Ошибка: {e}")
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None
except OSError as e:
    if "libXrender.so.1" in str(e) or "libX11" in str(e) or "X11" in str(e):
        logger.warning("rdkit установлен, но отсутствуют X11 библиотеки. 3D-визуализация будет ограничена.")
        logger.info("Для решения установите: sudo apt-get install libxrender1 libx11-6 libxext6")
        RDKIT_AVAILABLE = False
        Chem = None
        AllChem = None
    else:
        logger.warning(f"rdkit недоступен из-за системной ошибки: {e}")
        RDKIT_AVAILABLE = False
        Chem = None
        AllChem = None
except Exception as e:
    logger.warning(f"Неожиданная ошибка при импорте rdkit: {e}")
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

# Попытка импорта Py3Dmol
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Py3Dmol не установлен. 3D-визуализация недоступна. {e}")
    PY3DMOL_AVAILABLE = False
    py3Dmol = None

# Директория для кэша структур
CACHE_DIR = Path("ui/cache/structures")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def generate_mol_hash(smiles: str) -> str:
    """Генерирует хэш для кэширования молекулы"""
    return hashlib.md5(smiles.encode()).hexdigest()


def cache_structure_file(smiles: str, pdb_data: str) -> str:
    """Кэширует PDB данные в файл"""
    mol_hash = generate_mol_hash(smiles)
    cache_file = CACHE_DIR / f"{mol_hash}.pdb"

    with open(cache_file, 'w') as f:
        f.write(pdb_data)

    return str(cache_file)


def load_cached_structure(smiles: str) -> Optional[str]:
    """Загружает кэшированную структуру"""
    mol_hash = generate_mol_hash(smiles)
    cache_file = CACHE_DIR / f"{mol_hash}.pdb"

    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return f.read()

    return None


def smiles_to_3d(smiles: str) -> Optional[str]:
    """
    Конвертирует SMILES в 3D PDB структуру

    Args:
        smiles: SMILES строка молекулы

    Returns:
        PDB строка или None при ошибке
    """
    if not RDKIT_AVAILABLE:
        logger.error("rdkit недоступен для генерации 3D структур")
        return None

    try:
        # Проверяем кэш сначала
        cached = load_cached_structure(smiles)
        if cached:
            logger.info(f"Загружена кэшированная структура для {smiles}")
            return cached

        # Создаем молекулу из SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Предоставляем более понятное сообщение об ошибке
            common_abbreviations = {
                'ATP': 'Adenosine Triphosphate',
                'ADP': 'Adenosine Diphosphate',
                'GTP': 'Guanosine Triphosphate',
                'GDP': 'Guanosine Diphosphate',
                'NAD': 'Nicotinamide Adenine Dinucleotide',
                'NADH': 'Nicotinamide Adenine Dinucleotide (reduced)',
                'FAD': 'Flavin Adenine Dinucleotide',
                'FADH2': 'Flavin Adenine Dinucleotide (reduced)',
                'DNA': 'Deoxyribonucleic Acid',
                'RNA': 'Ribonucleic Acid'
            }

            if smiles.upper() in common_abbreviations:
                logger.error(f"Обнаружена аббревиатура вместо SMILES: {smiles} ({common_abbreviations[smiles.upper()]})")
            else:
                logger.error(f"Не удалось распарсить SMILES: {smiles}")
            return None

        # Проверяем, содержит ли SMILES явные водороды
        has_explicit_hydrogens = '[H]' in smiles

        if not has_explicit_hydrogens:
            # Добавляем водороды только если они не указаны явно
            mol = Chem.AddHs(mol)
            logger.info(f"Добавлены неявные водороды для {smiles}")
        else:
            # Для SMILES с явными водородами проверяем валидность
            logger.info(f"Обнаружены явные водороды в SMILES: {smiles}")

            # Проверяем, что все атомы имеют правильную валентность
            try:
                Chem.SanitizeMol(mol)
                logger.info(f"SMILES с явными водородами прошел валидацию: {smiles}")
            except Exception as e:
                logger.warning(f"Проблема с валентностью в SMILES: {smiles}, ошибка: {e}")
                # Пробуем исправить валентность
                try:
                    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
                    logger.info(f"SMILES исправлен: {smiles}")
                except Exception as e2:
                    logger.error(f"Не удалось исправить SMILES: {smiles}, ошибка: {e2}")
                    return None

        # Генерируем 3D координаты
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            logger.warning(f"Не удалось сгенерировать 3D структуру для {smiles}, пробуем альтернативные методы")

            # Пробуем с другими параметрами
            result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if result == -1:
                logger.warning(f"Повторная генерация не удалась, пробуем ETKDG метод")
                # Пробуем ETKDG метод (более надежный для сложных структур)
                try:
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    result = AllChem.EmbedMolecule(mol, params)
                    if result == -1:
                        logger.error(f"Все методы генерации 3D структуры не удались для {smiles}")
                        return None
                except Exception as e:
                    logger.error(f"Ошибка ETKDG метода для {smiles}: {e}")
                    return None

        # Оптимизируем геометрию
        try:
            optimize_result = AllChem.MMFFOptimizeMolecule(mol)
            if optimize_result == -1:
                logger.warning(f"Не удалось оптимизировать геометрию для {smiles}, используем без оптимизации")
            else:
                logger.info(f"Геометрия оптимизирована для {smiles}")
        except Exception as e:
            logger.warning(f"Ошибка оптимизации геометрии для {smiles}: {e}, продолжаем без оптимизации")

        # Конвертируем в PDB формат
        pdb_data = Chem.MolToPDBBlock(mol)

        # Проверяем, что PDB данные корректны
        if not pdb_data or len(pdb_data.strip()) < 50:
            logger.error(f"Сгенерированные PDB данные некорректны для {smiles}")
            return None

        # Кэшируем результат
        cache_structure_file(smiles, pdb_data)

        logger.info(f"Сгенерирована 3D структура для {smiles} ({len(pdb_data)} символов)")
        return pdb_data

    except Exception as e:
        logger.error(f"Ошибка генерации 3D структуры для {smiles}: {e}")
        return None


def create_3d_visualization(pdb_data: str, width: int = 600, height: int = 400) -> str:
    """
    Создает HTML компонент для 3D-визуализации

    Args:
        pdb_data: PDB данные молекулы
        width: Ширина виджета
        height: Высота виджета

    Returns:
        HTML строка для отображения
    """
    # Всегда используем альтернативную визуализацию вместо Py3Dmol
    # Py3Dmol требует IPython notebook, что несовместимо со Streamlit
    return create_3d_visualization_alternative(pdb_data, width, height)


def create_fallback_visualization(width: int = 600, height: int = 400, message: str = "3D визуализация недоступна") -> str:
    """
    Создает fallback визуализацию когда 3D не работает
    """
    return f"""
    <div style="width: {width}px; height: {height}px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: 2px solid #e9ecef; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="text-align: center; color: white; padding: 20px;">
            <div style="font-size: 48px; margin-bottom: 16px;">🧬</div>
            <h4 style="margin: 0 0 8px 0; font-weight: 600;">{message}</h4>
            <p style="margin: 0; opacity: 0.9;">Попробуйте 2D структуру ниже</p>
        </div>
    </div>
    """


def create_3d_visualization_alternative(pdb_data: str, width: int = 600, height: int = 400) -> str:
    """
    Альтернативная 3D визуализация без Py3Dmol - чистый HTML/CSS/JS с 3Dmol.js
    """
    try:
        # Проверяем, есть ли PDB данные
        if not pdb_data or len(pdb_data.strip()) == 0:
            return create_fallback_visualization(width, height, "Нет PDB данных")

        # Экранируем PDB данные для JavaScript
        escaped_pdb = pdb_data.replace('`', '\\`').replace('${', '\\${').replace('\\', '\\\\')

        # Проверяем длину PDB данных
        if len(escaped_pdb) > 10000:
            logger.warning(f"PDB данные слишком длинные ({len(escaped_pdb)} символов), могут вызвать проблемы с отображением")

        # Проверяем, что PDB данные содержат корректные заголовки
        if not (escaped_pdb.strip().startswith('HEADER') or
                escaped_pdb.strip().startswith('ATOM') or
                escaped_pdb.strip().startswith('HETATM') or
                'HETATM' in escaped_pdb[:200]):
            logger.warning(f"PDB данные могут быть некорректными: {escaped_pdb[:100]}...")
        else:
            logger.info(f"PDB данные содержат корректные записи (ATOM/HETATM/HEADER)")

        # Создаем HTML по частям для избежания проблем с f-строками
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>',
            '    <style>',
            '        #viewer {',
            '            width: ' + str(width) + 'px;',
            '            height: ' + str(height) + 'px;',
            '            position: relative;',
            '            border: 2px solid #e9ecef;',
            '            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);',
            '        }',
            '        .viewer-controls {',
            '            position: absolute;',
            '            top: 10px;',
            '            right: 10px;',
            '            z-index: 1000;',
            '            display: flex;',
            '            gap: 5px;',
            '        }',
            '        .control-btn {',
            '            background: rgba(255, 255, 255, 0.9);',
            '            border: 1px solid #dee2e6;',
            '            border-radius: 4px;',
            '            padding: 4px 8px;',
            '            font-size: 12px;',
            '            cursor: pointer;',
            '            transition: all 0.2s;',
            '        }',
            '        .control-btn:hover {',
            '            background: white;',
            '            box-shadow: 0 2px 4px rgba(0,0,0,0.1);',
            '        }',
            '    </style>',
            '</head>',
            '<body>',
            '    <div id="viewer">',
            '        <div class="viewer-controls">',
            '            <button class="control-btn" onclick="resetView()">Центр</button>',
            '            <button class="control-btn" onclick="toggleSpin()">Вращение</button>',
            '            <button class="control-btn" onclick="toggleStyle()">Стиль</button>',
            '        </div>',
            '    </div>',
            '',
            '    <script>',
            '        let viewer;',
            '        let isSpinning = false;',
            '        let currentStyle = \'stick\';',
            '',
            '        function initViewer() {',
            '            try {',
            '                viewer = $3Dmol.createViewer(\'viewer\', {',
            '                    defaultcolors: $3Dmol.rasmolColors',
            '                });',
            '',
            '                const pdbData = `' + escaped_pdb + '`;',
            '                console.log(\'PDB data length:\', pdbData.length);',
            '                console.log(\'PDB data preview:\', pdbData.substring(0, 200));',
            '',
            '                viewer.addModel(pdbData, \'pdb\');',
            '                console.log(\'Model added successfully\');',
            '',
            '                // Проверяем, что модель загружена',
            '                const atoms = viewer.getModel().selectedAtoms({});',
            '                console.log(\'Number of atoms loaded:\', atoms.length);',
            '',
            '                if (atoms.length === 0) {',
            '                    console.warn(\'No atoms loaded, trying alternative format\');',
            '                    viewer.clear();',
            '                    viewer.addModel(pdbData, \'mol\');',
            '                    console.log(\'Tried alternative mol format\');',
            '                }',
            '                viewer.setStyle({\'stick\': {\'colorscheme\': \'Jmol\'}});',
            '                viewer.zoomTo();',
            '                viewer.render();',
            '',
            '                console.log(\'3D модель загружена успешно\');',
            '            } catch (e) {',
            '                console.error(\'Ошибка загрузки PDB:\', e);',
            '                showError(\'Ошибка загрузки 3D структуры\');',
            '            }',
            '        }',
            '',
            '        function resetView() {',
            '            if (viewer) {',
            '                viewer.zoomTo();',
            '                viewer.render();',
            '            }',
            '        }',
            '',
            '        function toggleSpin() {',
            '            if (!viewer) return;',
            '            isSpinning = !isSpinning;',
            '            if (isSpinning) {',
            '                viewer.spin(\'y\', 0.01);',
            '            } else {',
            '                viewer.spin(false);',
            '            }',
            '        }',
            '',
            '        function toggleStyle() {',
            '            if (!viewer) return;',
            '            const styles = [\'stick\', \'sphere\'];',
            '            const currentIndex = currentStyle === \'stick\' ? 0 : 1;',
            '            currentStyle = styles[(currentIndex + 1) % styles.length];',
            '            ',
            '            viewer.setStyle({});',
            '            if (currentStyle === \'stick\') {',
            '                viewer.setStyle({\'stick\': {\'colorscheme\': \'Jmol\'}});',
            '            } else {',
            '                viewer.setStyle({\'sphere\': {\'colorscheme\': \'Jmol\'}});',
            '            }',
            '            viewer.render();',
            '        }',
            '',
            '        function showError(message) {',
            '            const viewerDiv = document.getElementById(\'viewer\');',
            '            viewerDiv.innerHTML = \'<div style="display: flex; align-items: center; justify-content: center; height: 100%; flex-direction: column; color: #dc3545; font-family: Arial, sans-serif;"><div style="font-size: 48px; margin-bottom: 16px;">⚠️</div><h4 style="margin: 0 0 8px 0;">Ошибка 3D визуализации</h4><p style="margin: 0; text-align: center;">\' + message + \'</p></div>\';',
            '        }',
            '',
            '        // Запуск',
            '        if (document.readyState === \'loading\') {',
            '            document.addEventListener(\'DOMContentLoaded\', initViewer);',
            '        } else {',
            '            initViewer();',
            '        }',
            '    </script>',
            '</body>',
            '</html>'
        ]

        html = '\n'.join(html_parts)
        return html

    except Exception as e:
        logger.error(f"Ошибка создания альтернативной 3D-визуализации: {e}")
        return create_fallback_visualization(width, height, f"Ошибка обработки PDB: {str(e)}")


def render_3d_structure(smiles: str, title: str = "3D Структура молекулы",
                        width: int = 600, height: int = 400) -> None:
    """
    Отображает 3D структуру молекулы в Streamlit с адаптивными размерами

    Args:
        smiles: SMILES строка молекулы
        title: Заголовок компонента
        width: Ширина виджета (адаптивная)
        height: Высота виджета (адаптивная)
    """
    if not smiles or smiles.strip() == "":
        st.warning("SMILES строка не указана")
        return

    st.subheader(f"🧬 {title}")

    # Индикатор загрузки
    with st.spinner("Генерация структуры..."):
        pdb_data = smiles_to_3d(smiles.strip())

        # Отладочная информация
        if pdb_data:
            st.info(f"✅ Успешно сгенерирована 3D структура для SMILES: {smiles}")
            st.text(f"PDB данные ({len(pdb_data)} символов):")
            with st.expander("Показать PDB данные", expanded=False):
                st.code(pdb_data[:1000] + "..." if len(pdb_data) > 1000 else pdb_data)
        else:
            st.warning(f"❌ Не удалось сгенерировать 3D структуру для SMILES: {smiles}")

    if pdb_data:
        # Информация для отладки
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🐛 Отладка", help="Показать отладочную информацию"):
                st.info("**Отладочная информация:**\n"
                       "- Откройте консоль браузера (F12)\n"
                       "- Ищите сообщения о PDB данных\n"
                       "- Проверьте ошибки JavaScript")

        # Адаптивные размеры для 3D визуализации
        container_width = min(width, 800)  # Максимум 800px
        container_height = min(height, 500)  # Максимум 500px
        html_content = create_3d_visualization_alternative(
            pdb_data, 
            width=container_width, 
            height=container_height
        )
        
        # Отображаем 3D визуализацию в адаптивном контейнере
        try:
            components.html(html_content, height=container_height, width=container_width, scrolling=False)
        except Exception as e:
            logger.error(f"Ошибка отображения 3D визуализации: {e}")
            st.error("❌ Ошибка отображения 3D визуализации")

            # Резервный вариант - показать PDB данные в текстовом виде
            st.subheader("📄 PDB данные (резервный вариант)")
            st.code(pdb_data, language="text")

            st.info("💡 **Совет:** Попробуйте другой SMILES или проверьте консоль браузера (F12) для отладочной информации")




    else:
        # Предоставляем более понятное сообщение об ошибке
        common_abbreviations = {
            'ATP': 'Adenosine Triphosphate',
            'ADP': 'Adenosine Diphosphate',
            'GTP': 'Guanosine Triphosphate',
            'GDP': 'Guanosine Diphosphate',
            'NAD': 'Nicotinamide Adenine Dinucleotide',
            'NADH': 'Nicotinamide Adenine Dinucleotide (reduced)',
            'FAD': 'Flavin Adenine Dinucleotide',
            'FADH2': 'Flavin Adenine Dinucleotide (reduced)',
            'DNA': 'Deoxyribonucleic Acid',
            'RNA': 'Ribonucleic Acid'
        }

        if smiles.upper() in common_abbreviations:
            st.error(f"❌ '{smiles}' - это аббревиатура ({common_abbreviations[smiles.upper()]}), а не SMILES строка")
            st.info("💡 **Подсказка:** Введите правильную SMILES строку, например: 'CC(=O)O' для уксусной кислоты")
        else:
            st.error(f"❌ Не удалось распарсить SMILES строку: '{smiles}'")
            st.info("💡 **Подсказка:** Проверьте корректность SMILES строки. Примеры: 'CC(=O)O', 'C1CCCCC1', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'")

        # Показываем 2D структуру если доступно
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol:
                    st.subheader("🖼️ 2D Структура (альтернатива)")
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        # Адаптивный размер изображения для альтернативного режима
                        img_size = min(300, 250)  # Компактный размер
                        img = Draw.MolToImage(mol, size=(img_size, img_size))
                        st.image(img, caption="2D структура молекулы", width='stretch')

                    with col2:
                        st.markdown("**Свойства молекулы:**")
                        st.info(f"""
                        • Атомы: {mol.GetNumAtoms()}
                        • Связи: {mol.GetNumBonds()}
                        • Молекулярная масса: {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f} Da
                        • Формула: {Chem.rdMolDescriptors.CalcMolFormula(mol)}
                        """)
            except Exception as e:
                logger.error(f"Ошибка генерации 2D структуры: {e}")

        # Предложения по исправлению
        with st.expander("💡 Советы по исправлению"):
            st.markdown("""
            **Возможные причины ошибки:**
            - Недопустимая SMILES строка
            - Слишком сложная молекула
            - Проблемы с генерацией 3D координат

            **Рекомендации:**
            - Проверьте корректность SMILES
            - Попробуйте упростить структуру
            - Используйте альтернативные SMILES нотации
            - Попробуйте Three.js визуализацию
            """)


def get_molecule_info(smiles: str) -> Dict[str, Any]:
    """
    Получает информацию о молекуле

    Args:
        smiles: SMILES строка

    Returns:
        Словарь с информацией о молекуле
    """
    if not RDKIT_AVAILABLE:
        return {"error": "rdkit недоступен"}

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return {"error": "Недопустимая SMILES строка"}

        info = {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "molecular_weight": Chem.rdMolDescriptors.CalcExactMolWt(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
            "logp": Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
            "tpsa": Chem.rdMolDescriptors.CalcTPSA(mol),
        }

        return info

    except Exception as e:
        return {"error": str(e)}


def display_molecule_properties(smiles: str) -> None:
    """
    Отображает свойства молекулы

    Args:
        smiles: SMILES строка молекулы
    """
    st.subheader("📊 Свойства молекулы")

    info = get_molecule_info(smiles)

    if "error" in info:
        st.error(f"Ошибка анализа: {info['error']}")
        return

    # Отображаем свойства в колонках
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Количество атомов", info["num_atoms"])
        st.metric("Молекулярная масса", f"{info['molecular_weight']:.2f} Da")

    with col2:
        st.metric("Количество связей", info["num_bonds"])
        st.metric("Количество колец", info["num_rings"])

    with col3:
        st.metric("LogP", f"{info['logp']:.2f}")
        st.metric("TPSA", f"{info['tpsa']:.1f} Å²")

    # Дополнительная информация
    with st.expander("📋 Детальная информация"):
        st.json(info)


def render_2d_structure(smiles: str, title: str = "2D Структура молекулы") -> None:
    """
    Улучшенная 2D визуализация молекулы с красивым дизайном

    Args:
        smiles: SMILES строка молекулы
        title: Заголовок компонента
    """
    if not smiles or smiles.strip() == "":
        st.warning("SMILES строка не указана")
        return

    if not RDKIT_AVAILABLE:
        st.error("❌ rdkit не установлен. Установите rdkit для 2D визуализации")
        return

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            st.error("❌ Не удалось распарсить SMILES строку")
            return

        # Создаем изображение с адаптивными размерами
        # Определяем оптимальный размер в зависимости от устройства
        optimal_size = min(500, 400)  # Максимум 500px, оптимум 400px
        img = Draw.MolToImage(mol, size=(optimal_size, optimal_size))

        # Отображаем в красивых контейнерах
        col1, col2 = st.columns([2, 1])

        with col1:

            st.image(img, caption="2D структура молекулы", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:

            st.markdown("### 📊 Основные свойства")
            
            # Основные метрики в красивых карточках
            col_a, col_b = st.columns(2)
            
            with col_a:
                
                st.markdown(f'<div class="structure-2d-metric-value">{mol.GetNumAtoms()}</div>', unsafe_allow_html=True)
                st.markdown('<div class="structure-2d-metric-label">Атомы</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                
                st.markdown(f'<div class="structure-2d-metric-value">{mol.GetNumBonds()}</div>', unsafe_allow_html=True)
                st.markdown('<div class="structure-2d-metric-label">Связи</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                
                st.markdown(f'<div class="structure-2d-metric-value">{Chem.rdMolDescriptors.CalcNumRings(mol)}</div>', unsafe_allow_html=True)
                st.markdown('<div class="structure-2d-metric-label">Кольца</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                
                st.markdown(f'<div class="structure-2d-metric-value">{Chem.rdMolDescriptors.CalcExactMolWt(mol):.1f}</div>', unsafe_allow_html=True)
                st.markdown('<div class="structure-2d-metric-label">Масса (Da)</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Дополнительная информация
        
        st.markdown("### 🔬 Детальная информация")
        
        # Дополнительные свойства
        logp = round(Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0], 2)
        tpsa = round(Chem.rdMolDescriptors.CalcTPSA(mol), 1)
        aromatic_rings = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
        rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        st.markdown(f"**🧪 Химическая формула:** `{Chem.rdMolDescriptors.CalcMolFormula(mol)}`")
        st.markdown(f"**📊 LogP:** {logp}")
        st.markdown(f"**📐 TPSA:** {tpsa} Å²")
        st.markdown(f"**💍 Ароматические кольца:** {aromatic_rings}")
        st.markdown(f"**🔄 Поворотные связи:** {rotatable_bonds}")
        
        # Определяем тип молекулы
        if aromatic_rings > 0:
            mol_type = "Ароматическая"
        elif Chem.rdMolDescriptors.CalcNumRings(mol) > 0:
            mol_type = "Циклическая"
        else:
            mol_type = "Алифатическая"
        
        st.markdown(f"**🏷️ Тип молекулы:** {mol_type}")
        
        # Оценка растворимости
        if logp < 0:
            solubility = "Хорошо растворима в воде"
        elif logp < 3:
            solubility = "Умеренно растворима"
        else:
            solubility = "Плохо растворима в воде"
        
        st.markdown(f"**💧 Растворимость:** {solubility}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Ошибка генерации 2D структуры: {e}")
        logger.error(f"Ошибка 2D визуализации для {smiles}: {e}")


def check_dependencies() -> Dict[str, bool]:
    """
    Проверяет доступность зависимостей

    Returns:
        Словарь со статусом зависимостей
    """
    return {
        "rdkit": RDKIT_AVAILABLE,
        "py3dmol": PY3DMOL_AVAILABLE,
    }


def install_instructions() -> str:
    """
    Возвращает инструкции по установке зависимостей
    """
    instructions = """
    ## 📦 Установка зависимостей для 3D-визуализации

    ### 1. rdkit (для генерации структур)
    ```bash
    conda install -c conda-forge rdkit
    # или
    pip install rdkit-pypi
    ```

    ### 2. Системные зависимости для rdkit (Linux)
    ```bash
    # Ubuntu/Debian
    sudo apt-get install libxrender1 libx11-6 libxext6 libxss1 libxrandr2
    
    # CentOS/RHEL/Fedora
    sudo yum install libXrender libX11 libXext libXScrnSaver libXrandr
    # или для новых версий
    sudo dnf install libXrender libX11 libXext libXScrnSaver libXrandr
    ```

    ### 3. Py3Dmol (для 3D-визуализации)
    ```bash
    pip install py3Dmol
    ```

    ### 4. Дополнительные зависимости
    ```bash
    pip install ipywidgets
    ```

    ### Проверка установки:
    ```python
    import rdkit
    import py3Dmol
    print("Все зависимости установлены!")
    ```

    ### Решение проблем с X11:
    Если rdkit установлен, но выдает ошибку "libXrender.so.1: cannot open shared object file",
    установите системные зависимости из пункта 2.
    """

    return instructions


# ===========================
# Продвинутые функции визуализации
# ===========================

def render_advanced_visualization_interface():
    """Интерфейс продвинутой визуализации с редактированием структур"""
    st.header("🎨 Продвинутая молекулярная визуализация")

    # Проверяем зависимости
    deps = check_dependencies()

    if not deps.get("rdkit", False):
        st.warning("⚠️ Для продвинутой визуализации требуется установка RDKit")
        st.code(install_instructions(), language="bash")
        return

    # Основные настройки визуализации
    col1, col2, col3 = st.columns(3)

    with col1:
        vis_mode = st.selectbox(
            "Режим визуализации:",
            ["3D структура", "2D структура", "Overlay сравнение", "Анимация"],
            help="Выберите тип визуализации"
        )

    with col2:
        style_options = ["stick", "sphere", "line", "cartoon", "surface"]
        vis_style = st.selectbox(
            "Стиль отображения:",
            style_options,
            index=0,
            help="Стиль отображения молекулы"
        )

    with col3:
        color_scheme = st.selectbox(
            "Цветовая схема:",
            ["default", "by element", "by residue", "rainbow", "chain"],
            help="Цветовая схема для атомов"
        )

    # Ввод SMILES для визуализации
    st.subheader("📝 Введите SMILES строку")

    smiles_input = st.text_input(
        "SMILES:",
        placeholder="Например: CC(=O)O (уксусная кислота) или C1CCCCC1 (циклогексан)",
        help="Введите SMILES строку молекулы для визуализации"
    )

    # Дополнительные опции в зависимости от режима
    if vis_mode == "Overlay сравнение":
        st.subheader("🔄 Сравнение структур")

        smiles2 = st.text_input(
            "SMILES второй молекулы:",
            placeholder="Введите SMILES второй молекулы для сравнения",
            help="Для overlay режима нужны две молекулы"
        )

        if smiles2:
            smiles_list = [smiles_input, smiles2] if smiles_input else [smiles2]
        else:
            smiles_list = [smiles_input] if smiles_input else []

    elif vis_mode == "Анимация":
        st.subheader("🎬 Настройки анимации")

        animation_type = st.selectbox(
            "Тип анимации:",
            ["вращение", "вибрация", "конформации"],
            help="Выберите тип анимации"
        )

        if animation_type == "конформации":
            n_conformers = st.slider("Количество конформеров:", 2, 10, 3)
        else:
            n_conformers = 1

        smiles_list = [smiles_input] if smiles_input else []

    else:
        smiles_list = [smiles_input] if smiles_input else []

    # Кнопка визуализации
    if st.button("🎨 Визуализировать", type="primary", width='stretch'):
        if not smiles_list or not any(smiles_list):
            st.error("❌ Введите хотя бы одну SMILES строку")
            return

        with st.spinner("Генерирую визуализацию..."):
            try:
                if vis_mode == "3D структура":
                    render_3d_structure(smiles_input)

                elif vis_mode == "2D структура":
                    render_2d_structure(smiles_input)

                elif vis_mode == "Overlay сравнение" and len(smiles_list) >= 2:
                    render_overlay_comparison(smiles_list[0], smiles_list[1], style=vis_style)

                elif vis_mode == "Анимация":
                    render_animation(smiles_input, animation_type, n_conformers, style=vis_style)

                else:
                    st.error("❌ Выбранный режим визуализации не поддерживается")

            except Exception as e:
                st.error(f"❌ Ошибка визуализации: {str(e)}")
                logger.error(f"Visualization error: {e}")

    # Инструменты редактирования (если выбрана молекула)
    if smiles_input and RDKIT_AVAILABLE:
        st.divider()
        st.subheader("🛠️ Инструменты редактирования")

        render_editing_tools(smiles_input)


def render_2d_structure(smiles: str):
    """Отображение 2D структуры молекулы"""
    if not RDKIT_AVAILABLE:
        st.error("RDKit не установлен")
        return

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Неверный SMILES формат")
            return

        # Создаем изображение
        img = Draw.MolToImage(mol, size=(600, 400))

        # Отображаем изображение
        st.image(img, caption=f"2D структура: {smiles}", use_column_width=True)

        # Дополнительная информация
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Атомы", mol.GetNumAtoms())
        with col2:
            st.metric("Связи", mol.GetNumBonds())
        with col3:
            st.metric("Кольца", Chem.rdMolDescriptors.CalcNumRings(mol))

    except Exception as e:
        st.error(f"Ошибка создания 2D структуры: {str(e)}")


def render_overlay_comparison(smiles1: str, smiles2: str, style: str = "stick"):
    """Overlay сравнение двух структур"""
    if not RDKIT_AVAILABLE or not PY3DMOL_AVAILABLE:
        st.error("Требуется RDKit и Py3DMol")
        return

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            st.error("Неверный формат SMILES")
            return

        # Генерируем 3D координаты
        mol1 = Chem.AddHs(mol1)
        mol2 = Chem.AddHs(mol2)
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.EmbedMolecule(mol2, randomSeed=42)

        # Создаем viewer
        viewer = py3Dmol.view(width=800, height=600)

        # Добавляем первую молекулу (синяя)
        molblock1 = Chem.MolToMolBlock(mol1)
        viewer.addModel(molblock1, 'mol')
        viewer.setStyle({'model': 0}, {style: {'color': 'blue'}})

        # Добавляем вторую молекулу (красная)
        molblock2 = Chem.MolToMolBlock(mol2)
        viewer.addModel(molblock2, 'mol')
        viewer.setStyle({'model': 1}, {style: {'color': 'red'}})

        # Настройки отображения
        viewer.zoomTo()
        viewer.setBackgroundColor('white')

        # Отображаем
        viewer_html = viewer._make_html()
        components.html(viewer_html, height=650)

        # Информация о сравнении
        st.info("🔵 Синяя молекула | 🔴 Красная молекула")

    except Exception as e:
        st.error(f"Ошибка overlay сравнения: {str(e)}")


def render_animation(smiles: str, animation_type: str, n_conformers: int = 3, style: str = "stick"):
    """Анимированная визуализация"""
    if not RDKIT_AVAILABLE or not PY3DMOL_AVAILABLE:
        st.error("Требуется RDKit и Py3DMol")
        return

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Неверный формат SMILES")
            return

        mol = Chem.AddHs(mol)

        if animation_type == "вращение":
            # Простое вращение
            AllChem.EmbedMolecule(mol, randomSeed=42)

            viewer = py3Dmol.view(width=800, height=600)
            molblock = Chem.MolToMolBlock(mol)
            viewer.addModel(molblock, 'mol')
            viewer.setStyle({style: {}})

            # Добавляем вращение
            viewer.spin(True)
            viewer.setBackgroundColor('white')
            viewer.zoomTo()

            viewer_html = viewer._make_html()
            components.html(viewer_html, height=650)

        elif animation_type == "конформации":
            # Генерируем несколько конформеров
            conformers = []
            try:
                AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, randomSeed=42)

                for i in range(n_conformers):
                    mol_copy = Chem.Mol(mol)
                    mol_copy.RemoveAllConformers()
                    mol_copy.AddConformer(mol.GetConformer(i))
                    conformers.append(mol_copy)

            except:
                # Если генерация конформеров не удалась, используем один
                AllChem.EmbedMolecule(mol, randomSeed=42)
                conformers = [mol]

            # Создаем анимацию
            viewer = py3Dmol.view(width=800, height=600)

            for i, conf_mol in enumerate(conformers):
                molblock = Chem.MolToMolBlock(conf_mol)
                viewer.addModel(molblock, 'mol')

            # Настройки анимации
            viewer.setStyle({style: {}})
            viewer.animate({'loop': 'backAndForth'})
            viewer.setBackgroundColor('white')
            viewer.zoomTo()

            viewer_html = viewer._make_html()
            components.html(viewer_html, height=650)

            st.info(f"🎬 Анимация {len(conformers)} конформеров")

        elif animation_type == "вибрация":
            # Имитация вибрации связей
            AllChem.EmbedMolecule(mol, randomSeed=42)

            viewer = py3Dmol.view(width=800, height=600)
            molblock = Chem.MolToMolBlock(mol)
            viewer.addModel(molblock, 'mol')
            viewer.setStyle({style: {}})

            # Добавляем вибрацию
            viewer.vibrate(0.5, 1.0)
            viewer.setBackgroundColor('white')
            viewer.zoomTo()

            viewer_html = viewer._make_html()
            components.html(viewer_html, height=650)

    except Exception as e:
        st.error(f"Ошибка анимации: {str(e)}")


def render_editing_tools(smiles: str):
    """Инструменты редактирования молекулы"""
    if not RDKIT_AVAILABLE:
        st.error("Требуется RDKit для инструментов редактирования")
        return

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return

        # Инструменты редактирования
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("⚛️ Добавить водороды", width='stretch'):
                mol_h = Chem.AddHs(mol)
                smiles_h = Chem.MolToSmiles(mol_h)
                st.code(f"SMILES с водородами:\n{smiles_h}")

        with col2:
            if st.button("🧹 Удалить стереохимию", width='stretch'):
                mol_clean = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
                if mol_clean:
                    smiles_clean = Chem.MolToSmiles(mol_clean, isomericSmiles=False)
                    st.code(f"SMILES без стереохимии:\n{smiles_clean}")

        with col3:
            if st.button("🔄 Канонизировать", width='stretch'):
                canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
                st.code(f"Канонический SMILES:\n{canonical_smiles}")

        with col4:
            if st.button("📊 Свойства", width='stretch'):
                # Отображаем свойства в expander
                with st.expander("Молекулярные свойства", expanded=True):
                    props = calculate_molecular_properties(smiles)
                    if props:
                        for prop, value in props.items():
                            st.write(f"**{prop}:** {value}")

    except Exception as e:
        st.error(f"Ошибка инструментов редактирования: {str(e)}")


def calculate_molecular_properties(smiles: str) -> Dict[str, Any]:
    """Расчет молекулярных свойств для продвинутой визуализации"""
    if not RDKIT_AVAILABLE:
        return {}

    try:
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        properties = {
            "Формула": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "Молекулярная масса": ".3f",
            "LogP": ".2f",
            "TPSA": ".2f",
            "Количество атомов": mol.GetNumAtoms(),
            "Количество тяжелых атомов": Chem.rdMolDescriptors.CalcNumHeavyAtoms(mol),
            "Количество акцепторов H": Chem.rdMolDescriptors.CalcNumHBA(mol),
            "Количество доноров H": Chem.rdMolDescriptors.CalcNumHBD(mol),
            "Количество ротамеров": Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Количество колец": Chem.rdMolDescriptors.CalcNumRings(mol),
        }

        return properties

    except Exception as e:
        logger.error(f"Ошибка расчета свойств: {e}")
        return {}
