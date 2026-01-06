import streamlit as st

try:
    import categorical_core
    st.success("✅ Rust-модуль работает!")
    st.write(categorical_core.hello())
except Exception as e:
    st.error(f"❌ Ошибка: {e}")
    st.code("Выполните в терминале: pip install -e .")
