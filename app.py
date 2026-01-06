# app.py
import streamlit as st
import pandas as pd
import json
import ollama
from io import StringIO
import time

# Проверка Rust-модуля
try:
    import categorical_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Настройка страницы
st.set_page_config(
    page_title="Когнитивный анализатор",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("🧠 Когнитивный анализатор: теория категорий в действии")

if not RUST_AVAILABLE:
    st.error("❌ Rust-модуль не найден. Выполните в терминале: `pip install -e .`")
    st.stop()

# Инициализация состояния
if 'hierarchy' not in st.session_state:
    st.session_state.hierarchy = categorical_core.PyHierarchy()
if 'current_level' not in st.session_state:
    st.session_state.current_level = 0
if 'semantic_names' not in st.session_state:
    st.session_state.semantic_names = {}
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False

# ========== 1. Загрузка данных ==========
st.header("1. Загрузите данные")

col1, col2 = st.columns(2)
with col1:
    objects_file = st.file_uploader("Файл объектов (CSV)", type=["csv"])
with col2:
    relations_file = st.file_uploader("Файл связей (CSV)", type=["csv"])

# Кнопка примера
if st.button("✨ Использовать пример"):
    objects_file = StringIO("""id,type,properties.intensity,properties.cost
stress,state,0.8,
shopping,activity,,100.0
meditation,activity,,
work,context,,""")
    relations_file = StringIO("""source,target,relation_type,strength,timestamp
stress,shopping,triggers,0.8,1700000000.0
shopping,stress,increases,0.6,1700000000.0
shopping,meditation,leads_to,0.7,1700086400.0
stress,meditation,avoids,0.4,1700086400.0""")

# Обработка загрузки
if objects_file and relations_file:
    try:
        df_objects = pd.read_csv(objects_file)
        df_relations = pd.read_csv(relations_file)

        # Обработка свойств
        prop_cols = [col for col in df_objects.columns if col.startswith("properties.")]
        if prop_cols:
            df_objects["properties"] = df_objects[prop_cols].apply(
                lambda row: {
                    k.replace("properties.", ""): float(v) if pd.notna(v) else 0.0
                    for k, v in row.items() if pd.notna(v)
                },
                axis=1
            )
            df_objects = df_objects.drop(columns=prop_cols)

        # Подготовка объектов
        objects_batch = []
        for _, r in df_objects.iterrows():
            props = r.get("properties", {})
            if not isinstance(props, dict):
                props = {}
            objects_batch.append({
                "id": str(r["id"]),
                "type": str(r["type"]),
                "properties": props
            })

        # Подготовка морфизмов
        morphisms_batch = []
        for _, r in df_relations.iterrows():
            morph_data = {
                "source": str(r["source"]),
                "target": str(r["target"]),
                "type": str(r["relation_type"]),
                "strength": float(r.get("strength", 0.5)),
                "evidence": [],
            }
            if "timestamp" in r and pd.notna(r["timestamp"]):
                morph_data["timestamp"] = float(r["timestamp"])
            morphisms_batch.append(morph_data)

        # Создание базовой категории
        base_cat = categorical_core.PyCategory("level_0")
        base_cat.clear()
        base_cat.add_objects_batch(objects_batch)
        base_cat.add_morphisms_batch(morphisms_batch)

        # Загрузка в иерархию
        st.session_state.hierarchy.set_base_category(base_cat)
        st.session_state.current_level = 0
        st.success(f"✅ Загружено: {len(objects_batch)} объектов, {len(morphisms_batch)} связей")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

# ========== 2. Управление иерархией ==========
st.header("2. Иерархия абстракций")

try:
    levels = st.session_state.hierarchy.get_levels()
    if levels:
        st.write(f"Доступные уровни: {levels}")
        st.session_state.current_level = st.selectbox("Выберите уровень для анализа", levels)
except:
    levels = [0]

# Подъём на новый уровень
if levels:
    max_level = max(levels)
    col_lift1, col_lift2 = st.columns(2)
    with col_lift1:
        target_level = st.number_input("Новый уровень", min_value=max_level+1, value=max_level+1, step=1)
    with col_lift2:
        n_clusters = st.slider("Число кластеров", 2, 10, 3)

    if st.button("🔼 Поднять на уровень"):
        try:
            st.session_state.hierarchy.auto_lift(max_level, target_level, n_clusters)
            st.success(f"Уровень {target_level} создан!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Ошибка подъёма: {e}")

# ========== 3. Эндофункторы (временная динамика) ==========
st.header("3. Эндофункторы: динамика во времени")

has_timestamps = False
try:
    morphs = st.session_state.hierarchy.get_category_morphisms(0)
    has_timestamps = any(m.get("timestamp", 0) > 0 for m in morphs)
except:
    pass

if has_timestamps:
    st.info("⏱️ Обнаружены временные метки. Можно построить эндофунктор.")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_ts = st.number_input("Начало (Unix timestamp)", value=1700000000.0)
    with col_t2:
        end_ts = st.number_input("Конец (Unix timestamp)", value=1700086400.0)
    
    if st.button("🔄 Построить эндофунктор"):
        try:
            functor_name = st.session_state.hierarchy.build_temporal_endofunctor(0, start_ts, end_ts)
            st.session_state.current_endofunctor = functor_name
            st.success(f"Эндофунктор '{functor_name}' создан!")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
else:
    st.info("Загрузите данные с колонкой 'timestamp' для анализа динамики")

# ========== 4. Категория процессов ==========
st.header("4. Категория процессов и монады")

if st.button("🎲 Построить категорию процессов"):
    try:
        proc_name = st.session_state.hierarchy.build_process_category(st.session_state.current_level)
        st.session_state.process_category = proc_name
        st.success(f"Категория процессов '{proc_name}' создана!")
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

if 'process_category' in st.session_state:
    try:
        two_step = st.session_state.hierarchy.get_process_two_step(st.session_state.current_level + 2000)
        if two_step:
            st.subheader("Двухшаговые процессы")
            for proc in two_step:
                st.markdown(f"""
                **{proc['source']} → ... → {proc['target']}**  
                Вероятность: **{proc['probability']:.3f}**  
                Путь: `{proc['morphism_type']}`
                """)
    except Exception as e:
        st.error(f"Ошибка процессов: {e}")

# ========== 5. Естественные преобразования ==========
st.header("5. Естественные преобразования: сравнение моделей")

try:
    functor_names = st.session_state.hierarchy.get_functor_names()
    if len(functor_names) >= 2:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            f1 = st.selectbox("Функтор 1", functor_names)
        with col_f2:
            f2 = st.selectbox("Функтор 2", [f for f in functor_names if f != f1])
        
        if st.button("🔀 Сравнить функторы"):
            try:
                nt = st.session_state.hierarchy.compare_functors(f1, f2, 0)
                st.session_state.current_nt = nt
                st.success("Естественное преобразование построено!")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
    else:
        st.info("Создайте минимум два функтора для сравнения")
except:
    st.info("Нет функторов")

# Отображение естественного преобразования
if 'current_nt' in st.session_state:
    nt = st.session_state.current_nt
    st.subheader("Компоненты преобразования")
    try:
        components = nt.get_components()
        for obj_id, comp in components:
            delta = comp['strength']
            color = "#4CAF50" if delta > 0 else "#F44336"
            st.markdown(f"""
            **Объект `{obj_id}`**:  
            `{comp['source']}` ──({delta:.2f})──→ `{comp['target']}`
            """)
        
        if st.button("🔍 Проверить естественность"):
            is_nat = nt.is_natural(st.session_state.hierarchy)
            if is_nat:
                st.success("✅ Условие естественности выполнено!")
            else:
                st.warning("⚠️ Преобразование не является естественным")
    except Exception as e:
        st.error(f"Ошибка: {e}")

# ========== 6. Визуализация ==========
st.header("6. Визуализация")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🎨 Показать граф"):
        st.session_state.show_graph = True
with col_btn2:
    if st.button("🧹 Скрыть граф"):
        st.session_state.show_graph = False

if st.session_state.show_graph:
    try:
        from pyvis.network import Network
        import streamlit.components.v1 as components

        morphisms = st.session_state.hierarchy.get_category_morphisms(st.session_state.current_level)
        all_nodes = set()
        for m in morphisms:
            all_nodes.add(m["source"])
            all_nodes.add(m["target"])

        net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff")
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 50}
          },
          "edges": {"arrows": {"to": {"enabled": true}}}
        }
        """)

        for node in all_nodes:
            label = st.session_state.semantic_names.get(node, node)
            color = "#2196F3" if not node.startswith("L") else "#4CAF50"
            net.add_node(node, label=label, color=color, size=25)

        for m in morphisms:
            width = max(1, m["strength"] * 5)
            net.add_edge(
                m["source"], m["target"],
                title=f"Сила: {m['strength']:.2f}",
                width=width,
                color="#607D8B"
            )

        components.html(net.generate_html(), height=650)
    except ImportError:
        st.info("Установите pyvis: `pip install pyvis`")
    except Exception as e:
        st.error(f"Ошибка визуализации: {e}")

# ========== 7. Семантические имена ==========
st.header("7. Семантические имена (Qwen)")

ollama_available = True
try:
    ollama.list()
except:
    ollama_available = False

if not ollama_available:
    st.warning("⚠️ Запустите Ollama: `ollama run qwen2:0.5b`")
else:
    if st.button("🔤 Сгенерировать имена"):
        try:
            obj_ids = st.session_state.hierarchy.get_category_object_ids(st.session_state.current_level)
            names = {}
            progress = st.progress(0)
            status = st.empty()
            
            for i, obj in enumerate(obj_ids):
                status.text(f"Генерация для: {obj}")
                prompt = f"""
Вы — когнитивный лингвист. Объект "{obj}" участвует в поведенческих циклах.
Дайте этому объекту глубокое, смысловое название на русском языке (1-3 слова).
Пример: "shopping" → "компенсаторные покупки".
Ответ в формате JSON: {{"название": "..."}}
                """.strip()

                response = ollama.generate(
                    model="qwen2:0.5b",
                    prompt=prompt,
                    format="json",
                    options={"temperature": 0.3}
                )
                try:
                    result = json.loads(response["response"])
                    names[obj] = result.get("название", obj)
                except:
                    names[obj] = obj
                progress.progress((i + 1) / len(obj_ids))
            
            st.session_state.semantic_names.update(names)
            st.success("Имена сгенерированы!")
            status.empty()
        except Exception as e:
            st.error(f"Ошибка Qwen: {e}")

# Отображение имён
if st.session_state.semantic_names:
    st.subheader("Сгенерированные имена:")
    for obj, name in st.session_state.semantic_names.items():
        st.write(f"**{obj}** → {name}")

# ========== Инструкция ==========
st.markdown("---")
st.caption("""
**Как запустить локально**:  
```bash
git clone https://github.com/ваш-логин/categorical-hierarchy
cd categorical-hierarchy
pip install -e .
streamlit run app.py
