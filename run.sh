#!/bin/bash
set -e  # остановить при ошибке

echo "🔧 Устанавливаем Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "📦 Устанавливаем Python-зависимости..."
pip install -r requirements.txt
pip install -e .

echo "🚀 Запускаем приложение..."
streamlit run app.py --server.port=8080 --server.address=0.0.0.0
