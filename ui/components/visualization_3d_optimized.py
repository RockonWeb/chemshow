"""
Оптимизированная 3D-визуализация молекулярных структур
"""
import streamlit as st
import streamlit.components.v1 as components
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
import asyncio
from functools import lru_cache
import base64

logger = logging.getLogger(__name__)

# Ленивый импорт RDKit
_rdkit_loaded = False
_rdkit_modules = {}


def lazy_import_rdkit():
    """Ленивый импорт RDKit модулей"""
    global _rdkit_loaded, _rdkit_modules
    
    if _rdkit_loaded:
        return _rdkit_modules
    
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw
        
        _rdkit_modules = {
            'Chem': Chem,
            'AllChem': AllChem,
            'Draw': Draw
        }
        _rdkit_loaded = True
        logger.info("RDKit modules loaded successfully")
        
    except ImportError as e:
        logger.warning(f"RDKit not available: {e}")
        _rdkit_modules = {}
    
    return _rdkit_modules


# Кэш для структур с LRU стратегией
@lru_cache(maxsize=100)
def get_cached_structure(smiles: str) -> Optional[str]:
    """Получает кэшированную структуру"""
    cache_dir = Path("ui/cache/structures")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    mol_hash = hashlib.md5(smiles.encode()).hexdigest()
    cache_file = cache_dir / f"{mol_hash}.pdb"
    
    if cache_file.exists():
        return cache_file.read_text()
    
    return None


def save_cached_structure(smiles: str, pdb_data: str) -> None:
    """Сохраняет структуру в кэш"""
    cache_dir = Path("ui/cache/structures")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    mol_hash = hashlib.md5(smiles.encode()).hexdigest()
    cache_file = cache_dir / f"{mol_hash}.pdb"
    
    cache_file.write_text(pdb_data)


@st.cache_data(ttl=3600)
def generate_3d_structure_optimized(smiles: str) -> Optional[str]:
    """Оптимизированная генерация 3D структуры с кэшированием"""
    
    # Проверяем кэш
    cached = get_cached_structure(smiles)
    if cached:
        logger.info(f"Structure loaded from cache for {smiles}")
        return cached
    
    # Импортируем RDKit только при необходимости
    rdkit = lazy_import_rdkit()
    if not rdkit or 'Chem' not in rdkit:
        return None
    
    Chem = rdkit['Chem']
    AllChem = rdkit['AllChem']
    
    try:
        # Парсим SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES: {smiles}")
            return None
        
        # Добавляем водороды если нужно
        if '[H]' not in smiles:
            mol = Chem.AddHs(mol)
        
        # Генерируем 3D координаты с оптимизированными параметрами
        # Используем ETKDG метод который быстрее и надежнее
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 0  # Используем все доступные потоки
        
        result = AllChem.EmbedMolecule(mol, params)
        
        if result == -1:
            # Пробуем упрощенный метод
            result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if result == -1:
                logger.error(f"Failed to generate 3D coordinates for {smiles}")
                return None
        
        # Быстрая оптимизация геометрии (меньше итераций)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        
        # Конвертируем в PDB
        pdb_data = Chem.MolToPDBBlock(mol)
        
        # Сохраняем в кэш
        save_cached_structure(smiles, pdb_data)
        
        logger.info(f"3D structure generated for {smiles}")
        return pdb_data
        
    except Exception as e:
        logger.error(f"Error generating 3D structure: {e}")
        return None


def create_optimized_3d_viewer(pdb_data: str, width: int = 600, height: int = 400) -> str:
    """Создает оптимизированный 3D viewer с минимальным JavaScript"""
    
    # Кодируем PDB данные в base64 для безопасной передачи
    pdb_b64 = base64.b64encode(pdb_data.encode()).decode()
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js" defer></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
            #viewer {{
                width: {width}px;
                height: {height}px;
                position: relative;
            }}
            .controls {{
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 1000;
            }}
            .btn {{
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px 10px;
                margin: 2px;
                cursor: pointer;
                font-size: 12px;
            }}
            .btn:hover {{
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .loading {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-family: Arial, sans-serif;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div id="viewer">
            <div class="loading">Loading 3D structure...</div>
            <div class="controls" style="display: none;">
                <button class="btn" onclick="resetView()">Reset</button>
                <button class="btn" onclick="toggleSpin()">Spin</button>
                <button class="btn" onclick="changeStyle()">Style</button>
            </div>
        </div>
        
        <script>
        // Декодируем PDB данные
        const pdbData = atob('{pdb_b64}');
        
        let viewer = null;
        let spinning = false;
        let currentStyle = 0;
        const styles = ['stick', 'sphere', 'cartoon'];
        
        // Инициализация после загрузки библиотеки
        function initViewer() {{
            if (typeof $3Dmol === 'undefined') {{
                setTimeout(initViewer, 100);
                return;
            }}
            
            viewer = $3Dmol.createViewer('viewer', {{
                defaultcolors: $3Dmol.rasmolColors,
                backgroundColor: 'white'
            }});
            
            viewer.addModel(pdbData, 'pdb');
            viewer.setStyle({{}}, {{stick: {{colorscheme: 'Jmol'}}}});
            viewer.zoomTo();
            viewer.render();
            
            // Показываем контролы
            document.querySelector('.loading').style.display = 'none';
            document.querySelector('.controls').style.display = 'block';
        }}
        
        function resetView() {{
            if (viewer) {{
                viewer.zoomTo();
                viewer.render();
            }}
        }}
        
        function toggleSpin() {{
            if (viewer) {{
                spinning = !spinning;
                viewer.spin(spinning ? 'y' : false);
            }}
        }}
        
        function changeStyle() {{
            if (viewer) {{
                currentStyle = (currentStyle + 1) % styles.length;
                const style = styles[currentStyle];
                viewer.setStyle({{}}, {{[style]: {{colorscheme: 'Jmol'}}}});
                viewer.render();
            }}
        }}
        
        // Запуск инициализации
        document.addEventListener('DOMContentLoaded', initViewer);
        </script>
    </body>
    </html>
    """
    
    return html


@st.cache_data(ttl=3600)
def generate_2d_structure_optimized(smiles: str, size: Tuple[int, int] = (350, 350)) -> Optional[bytes]:
    """Оптимизированная генерация 2D структуры"""
    
    rdkit = lazy_import_rdkit()
    if not rdkit or 'Chem' not in rdkit:
        return None
    
    Chem = rdkit['Chem']
    Draw = rdkit['Draw']
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Генерируем изображение
        img = Draw.MolToImage(mol, size=size)
        
        # Конвертируем в байты
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating 2D structure: {e}")
        return None


def render_3d_structure_optimized(
    smiles: str,
    title: str = "3D Structure",
    width: int = 600,
    height: int = 400,
    key: str = None
) -> None:
    """Оптимизированный рендеринг 3D структуры"""
    
    if not smiles or not smiles.strip():
        st.warning("No SMILES string provided")
        return
    
    # Используем уникальный ключ для кэширования в session state
    cache_key = f"3d_structure_{hashlib.md5(smiles.encode()).hexdigest()}"
    
    # Проверяем session state кэш
    if cache_key in st.session_state:
        pdb_data = st.session_state[cache_key]
    else:
        # Генерируем структуру с прогресс-баром
        with st.spinner("Generating 3D structure..."):
            pdb_data = generate_3d_structure_optimized(smiles.strip())
            if pdb_data:
                st.session_state[cache_key] = pdb_data
    
    if pdb_data:
        st.subheader(f"🧬 {title}")
        
        # Создаем HTML с 3D viewer
        html_content = create_optimized_3d_viewer(pdb_data, width, height)
        
        # Отображаем компонент
        components.html(html_content, height=height + 50, width=width)
        
        # Информация о молекуле
        with st.expander("📊 Molecule Info"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**SMILES:** `{smiles}`")
                
                # Быстрый подсчет атомов
                rdkit = lazy_import_rdkit()
                if rdkit and 'Chem' in rdkit:
                    mol = rdkit['Chem'].MolFromSmiles(smiles)
                    if mol:
                        st.write(f"**Atoms:** {mol.GetNumAtoms()}")
                        st.write(f"**Bonds:** {mol.GetNumBonds()}")
            
            with col2:
                # 2D структура как превью
                img_bytes = generate_2d_structure_optimized(smiles, size=(200, 200))
                if img_bytes:
                    st.image(img_bytes, caption="2D Structure", width=200)
    else:
        st.error(f"Failed to generate 3D structure for: {smiles}")
        
        # Показываем 2D структуру как альтернативу
        st.subheader("🖼️ 2D Structure (Alternative)")
        img_bytes = generate_2d_structure_optimized(smiles)
        if img_bytes:
            st.image(img_bytes, caption="2D Structure")


def render_molecule_comparison_optimized(
    smiles1: str,
    smiles2: str,
    title: str = "Molecule Comparison"
) -> None:
    """Оптимизированное сравнение двух молекул"""
    
    st.subheader(f"🔬 {title}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Molecule 1**")
        img1 = generate_2d_structure_optimized(smiles1)
        if img1:
            st.image(img1, width=300)
        st.code(smiles1)
    
    with col2:
        st.write("**Molecule 2**")
        img2 = generate_2d_structure_optimized(smiles2)
        if img2:
            st.image(img2, width=300)
        st.code(smiles2)
    
    # Сравнение свойств
    rdkit = lazy_import_rdkit()
    if rdkit and 'Chem' in rdkit:
        Chem = rdkit['Chem']
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 and mol2:
            st.write("**Property Comparison:**")
            
            from rdkit.Chem import Descriptors
            
            props = {
                "Molecular Weight": (
                    Descriptors.MolWt(mol1),
                    Descriptors.MolWt(mol2)
                ),
                "LogP": (
                    Descriptors.MolLogP(mol1),
                    Descriptors.MolLogP(mol2)
                ),
                "TPSA": (
                    Descriptors.TPSA(mol1),
                    Descriptors.TPSA(mol2)
                ),
                "Rotatable Bonds": (
                    Descriptors.NumRotatableBonds(mol1),
                    Descriptors.NumRotatableBonds(mol2)
                )
            }
            
            import pandas as pd
            df = pd.DataFrame(props, index=["Molecule 1", "Molecule 2"]).T
            st.dataframe(df)


# Экспорт оптимизированных функций
__all__ = [
    'render_3d_structure_optimized',
    'generate_3d_structure_optimized',
    'generate_2d_structure_optimized',
    'render_molecule_comparison_optimized'
]
