import streamlit as st
import pandas as pd
import json
import os
import requests
from pyvis.network import Network
import tempfile

# ========== Настройка страницы ==========
st.set_page_config(
    page_title="🧠 Когнитивный анализатор",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Инициализация ==========
st.title("🧠 Когнитивный анализатор на основе теории категорий")

# Проверка Rust-модуля
RUST_AVAILABLE = False
try:
    import categorical_core
    RUST_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Модуль Rust не найден. Выполните в терминале: `pip install -e .`")
    st.code(str(e))
    st.stop()

# ========== Загрузка данных ==========
st.subheader("📥 Загрузите данные")

col1, col2 = st.columns([3, 1])
with col2:
    use_example = st.button("✨ Использовать пример")

if use_example:
    # Пример объектов
    objects_df = pd.DataFrame({
        "id": ["user1", "task1", "reward1"],
        "object_type": ["human", "task", "reward"],
        "properties.mood": [0.8, 0.5, 0.9],
        "properties.energy": [0.6, 0.9, 0.7]
    })
    # Пример морфизмов
    morphisms_df = pd.DataFrame({
        "source": ["user1", "user1", "task1"],
        "target": ["task1", "reward1", "reward1"],
        "morphism_type": ["performs", "seeks", "yields"],
        "strength": [0.75, 0.85, 0.65]
    })
else:
    st.info("Загрузите два CSV-файла: объекты и морфизмы.")
    obj_file = st.file_uploader("Файл объектов (с колонками: id, object_type, properties.*)", type="csv")
    morph_file = st.file_uploader("Файл морфизмов (с колонками: source, target, morphism_type, strength)", type="csv")
    
    if obj_file and morph_file:
        objects_df = pd.read_csv(obj_file)
        morphisms_df = pd.read_csv(morph_file)
    else:
        st.stop()

# ========== Анализ ==========
st.subheader("🔍 Анализ")

if st.button("🚀 Построить категорию и проанализировать"):
    try:
        # Создаём категорию
        cat = categorical_core.PyCategory("cognitive_model")
        
        # Добавляем объекты
        for _, row in objects_df.iterrows():
            props = {}
            for col in objects_df.columns:
                if col.startswith("properties."):
                    key = col.replace("properties.", "")
                    props[key] = float(row[col])
            cat.add_object(str(row["id"]), str(row["object_type"]), props)
        
        # Добавляем морфизмы
        for _, row in morphisms_df.iterrows():
            cat.add_morphism(
                str(row["source"]),
                str(row["target"]),
                str(row["morphism_type"]),
                float(row["strength"])
            )
        
        st.success(f"✅ Категория построена: {cat.object_count()} объектов, {cat.morphism_count()} морфизмов")
        
        # Анализ циклов
        cycles = cat.detect_cycles()
        st.write(f"🔄 Обнаружено циклов: {len(cycles)}")
        if cycles:
            for i, cycle in enumerate(cycles[:3]):
                st.write(f"Цикл {i+1}: {' → '.join(cycle)}")
        
        # Иерархия
        hierarchy = cat.build_hierarchy()
        st.write("📊 Иерархия уровней:")
        for level, nodes in enumerate(hierarchy):
            st.write(f"Уровень {level}: {', '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}")
        
        # Генерация имени
        try:
            summary = f"Категория с {cat.object_count()} объектами, {cat.morphism_count()} морфизмами, {len(cycles)} циклами."
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2:0.5b",
                    "prompt": f"Дай краткое семантическое имя этой когнитивной категории: {summary}",
                    "stream": False
                },
                timeout=10
            )
            if response.ok:
                name = response.json().get("response", "").strip()
                st.subheader(f"🏷️ Семантическое имя: {name}")
            else:
                st.warning("⚠️ Не удалось сгенерировать имя (Ollama не отвечает)")
        except Exception as e:
            st.warning(f"⚠️ Ollama недоступен: {str(e)}")
        
        # Визуализация
net = Network(
    height="700px",
    width="100%",
    directed=True,
    notebook=False,
    bgcolor="#ffffff",
    font_color="black"
)

# Настройка физики: отключаем "липкость"
net.set_options("""
var options = {
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 100},
    "barnesHut": {
      "gravitationalConstant": -10000,  // слабее притяжение
      "centralGravity": 0.1,
      "springLength": 200,             // больше расстояние между узлами
      "springConstant": 0.01,          // слабее "резинка"
      "damping": 0.5,
      "avoidOverlap": 1
    }
  },
  "interaction": {
    "dragNodes": true,                // можно тащить узлы
    "hover": true,
    "tooltipDelay": 200
  }
}
""")

# Добавляем узлы с размером по criticality
for _, row in st.session_state.objects_df.iterrows():
    size = 10 + 20 * float(row.get("properties.criticality", 0.5))
    net.add_node(
        row["id"],
        label=f"{row['id']}\n({row['object_type']})",
        size=size,
        color="#97c2fc" if row["object_type"] == "equipment" else 
               "#f4c28f" if row["object_type"] == "team" else
               "#a2d2a4" if row["object_type"] == "personnel" else
               "#e0e0e0"
    )

# Добавляем рёбра с толщиной по strength
for _, row in st.session_state.morphisms_df.iterrows():
    width = 1 + 5 * float(row["strength"])
    net.add_edge(
        row["source"],
        row["target"],
        label=row["morphism_type"],
        title=f"Strength: {row['strength']}",
        width=width,
        arrows="to"
    )

# Сохраняем и отображаем
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
    net.save_graph(f.name)
    with open(f.name) as g:
        html = g.read()
st.components.v1.html(html, height=700)

# ========== Инструкция ==========
st.markdown("---")
st.caption("""
**Как запустить локально**:  
```bash
git clone https://github.com/KostyaMartov/categorical-hierarchy
cd categorical-hierarchy
pip install -e .
streamlit run app.py
