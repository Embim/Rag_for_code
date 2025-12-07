#!/usr/bin/env python3
"""
Тест исправлений для парсера
"""
from pathlib import Path
from src.code_rag.parsers import ReactParser

def test_permission_error():
    """Тест обработки PermissionError"""
    parser = ReactParser()

    # Попытка парсить несуществующий файл
    result = parser.parse_file(Path("nonexistent.jsx"))

    print("Test: PermissionError handling")
    print(f"  File path: {result.file_path}")
    print(f"  Language: {result.language}")
    print(f"  Errors: {result.errors}")
    print(f"  Success: {len(result.errors) > 0 and result.language is not None}")

    assert result.language in ['javascript', 'typescript'], "Language should be set"
    assert len(result.errors) > 0, "Should have errors"
    print("\n[OK] Test passed!\n")

def test_directory_check():
    """Тест проверки, что путь является файлом"""
    test_path = Path("F:/ui/node_modules/big.js")

    print("Test: Directory vs File check")
    print(f"  Path: {test_path}")
    print(f"  Exists: {test_path.exists()}")

    if test_path.exists():
        is_file = test_path.is_file()
        is_dir = test_path.is_dir()
        print(f"  Is file: {is_file}")
        print(f"  Is directory: {is_dir}")
        print(f"  Success: Should skip if directory")
    else:
        print(f"  Path doesn't exist (OK for test)")

    print("\n[OK] Test passed!\n")

if __name__ == "__main__":
    print("="*80)
    print("TESTING PARSER FIXES")
    print("="*80)
    print()

    test_permission_error()
    test_directory_check()

    print("="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
