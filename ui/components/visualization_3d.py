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
import py3Dmol
# Настройка логирования
logger = logging.getLogger(__name__)

# Попытка импорта rdkit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("rdkit не установлен. 3D-визуализация будет ограничена.")
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

# Попытка импорта Py3Dmol
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    logger.warning("Py3Dmol не установлен. 3D-визуализация недоступна.")
    PY3DMOL_AVAILABLE = False
    py3Dmol = None


# Настройка логирования
logger = logging.getLogger(__name__)

# Попытка импорта rdkit
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    logger.warning("rdkit не установлен. 3D-визуализация будет ограничена.")
    RDKIT_AVAILABLE = False
    Chem = None
    AllChem = None

# Попытка импорта Py3Dmol
try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    logger.warning("Py3Dmol не установлен. 3D-визуализация недоступна.")
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
            logger.error(f"Не удалось распарсить SMILES: {smiles}")
            return None

        # Добавляем водороды
        mol = Chem.AddHs(mol)

        # Генерируем 3D координаты
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            logger.warning(f"Не удалось сгенерировать 3D структуру для {smiles}")
            # Пробуем с другими параметрами
            result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if result == -1:
                logger.error(f"Повторная генерация 3D структуры не удалась для {smiles}")
                return None

        # Оптимизируем геометрию
        AllChem.MMFFOptimizeMolecule(mol)

        # Конвертируем в PDB формат
        pdb_data = Chem.MolToPDBBlock(mol)

        # Кэшируем результат
        cache_structure_file(smiles, pdb_data)

        logger.info(f"Сгенерирована 3D структура для {smiles}")
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
        escaped_pdb = pdb_data.replace('`', '\\`').replace('${', '\\${')

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
            '',
            '                viewer.addModel(pdbData, \'pdb\');',
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

    if pdb_data:
        # Адаптивные размеры для 3D визуализации
        container_width = min(width, 800)  # Максимум 800px
        container_height = min(height, 500)  # Максимум 500px
        html_content = create_3d_visualization_alternative(
            pdb_data, 
            width=container_width, 
            height=container_height
        )
        
        # Отображаем 3D визуализацию в адаптивном контейнере
        components.html(html_content, height=container_height, width=container_width, scrolling=False)




    else:
        st.error("❌ Не удалось сгенерировать 3D структуру молекулы")

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
                        st.image(img, caption="2D структура молекулы", use_container_width=True)

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

            st.image(img, caption="2D структура молекулы", use_container_width=True)
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

    ### 2. Py3Dmol (для 3D-визуализации)
    ```bash
    pip install py3Dmol
    ```

    ### 3. Дополнительные зависимости
    ```bash
    pip install ipywidgets
    ```

    ### Проверка установки:
    ```python
    import rdkit
    import py3Dmol
    print("Все зависимости установлены!")
    ```
    """

    return instructions
