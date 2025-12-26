#!/usr/bin/env python3
"""
Тест для проверки загрузки LANGSMITH_API_KEY из корневого .env
"""
import os
import sys
from pathlib import Path

# Добавляем путь к проекту
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Импортируем rag_graph, который должен загрузить .env
from src.langgraph_server import rag_graph

# Проверяем, загружен ли LANGSMITH_API_KEY
langsmith_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_key:
    print(f"✅ LANGSMITH_API_KEY загружен: {langsmith_key[:20]}...")
    print(f"   Полный путь к .env: {PROJECT_ROOT / '.env'}")
else:
    print("❌ LANGSMITH_API_KEY не найден")
    print(f"   Проверьте файл: {PROJECT_ROOT / '.env'}")
    sys.exit(1)

