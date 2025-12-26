#!/usr/bin/env python3
"""
Скрипт для настройки .env файла для LangGraph в src/langgraph_server/

LangGraph ищет .env файл в директории запуска (src/langgraph_server/),
а не в корне проекта. Этот скрипт копирует LANGSMITH_API_KEY из корневого .env
в src/langgraph_server/.env
"""

import os
import shutil
from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
ROOT_ENV = PROJECT_ROOT / ".env"
LANGGRAPH_ENV = PROJECT_ROOT / "src" / "langgraph_server" / ".env"

def setup_langgraph_env():
    """Создать .env в src/langgraph_server/ с LANGSMITH_API_KEY."""
    
    if not ROOT_ENV.exists():
        print(f"❌ Файл {ROOT_ENV} не найден!")
        print("Создайте .env файл в корне проекта с LANGSMITH_API_KEY")
        return False
    
    # Читаем LANGSMITH_API_KEY из корневого .env
    langsmith_key = None
    with open(ROOT_ENV, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('LANGSMITH_API_KEY'):
                # Убираем пробелы и извлекаем значение
                if '=' in line:
                    key, value = line.split('=', 1)
                    langsmith_key = value.strip().strip('"').strip("'")
                    break
    
    if not langsmith_key:
        print("❌ LANGSMITH_API_KEY не найден в корневом .env файле")
        print("Добавьте строку: LANGSMITH_API_KEY=lsv2_pt_...")
        return False
    
    # Создаем .env в src/langgraph_server/
    LANGGRAPH_ENV.parent.mkdir(parents=True, exist_ok=True)
    
    # Читаем существующий .env если есть, или создаем новый
    existing_content = ""
    if LANGGRAPH_ENV.exists():
        with open(LANGGRAPH_ENV, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # Обновляем или добавляем LANGSMITH_API_KEY
    lines = existing_content.split('\n') if existing_content else []
    updated = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('LANGSMITH_API_KEY'):
            new_lines.append(f'LANGSMITH_API_KEY={langsmith_key}')
            updated = True
        else:
            new_lines.append(line)
    
    if not updated:
        new_lines.append(f'LANGSMITH_API_KEY={langsmith_key}')
    
    # Записываем обновленный файл
    with open(LANGGRAPH_ENV, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
        if not new_lines[-1]:  # Добавляем перенос строки в конце если нужно
            f.write('\n')
    
    print(f"✅ Создан/обновлен {LANGGRAPH_ENV}")
    print(f"   LANGSMITH_API_KEY установлен (первые 20 символов: {langsmith_key[:20]}...)")
    print("\nТеперь запустите: cd src/langgraph_server && langgraph dev")
    return True

if __name__ == "__main__":
    setup_langgraph_env()

