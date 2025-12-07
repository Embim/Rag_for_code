#!/usr/bin/env python3
"""
Проверка всех исправлений
"""
import sys
from pathlib import Path

def test_full_name_property():
    """Test 1: CodeEntity.full_name property"""
    print("Test 1: CodeEntity.full_name property")
    try:
        from src.code_rag.parsers.base import CodeEntity, EntityType

        # Test standalone entity
        entity1 = CodeEntity(name="my_function", type=EntityType.FUNCTION)
        assert entity1.full_name == "my_function", f"Expected 'my_function', got '{entity1.full_name}'"

        # Test method with parent
        entity2 = CodeEntity(name="save", type=EntityType.METHOD, parent="UserModel")
        assert entity2.full_name == "UserModel.save", f"Expected 'UserModel.save', got '{entity2.full_name}'"

        print("  [OK] full_name property works correctly\n")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}\n")
        return False


def test_react_parser_error_handling():
    """Test 2: React parser error handling"""
    print("Test 2: React parser error handling")
    try:
        from src.code_rag.parsers import ReactParser

        parser = ReactParser()

        # Test with non-existent file
        result = parser.parse_file(Path("nonexistent.jsx"))

        assert result.language in ['javascript', 'typescript'], \
            f"Language should be set, got: {result.language}"
        assert len(result.errors) > 0, "Should have errors"

        print("  [OK] Error handling works correctly\n")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}\n")
        return False


def test_subprocess_encoding():
    """Test 3: Check subprocess encoding is set"""
    print("Test 3: Subprocess encoding configuration")
    try:
        import subprocess
        from pathlib import Path

        # Read react_parser.py
        react_parser = Path(__file__).parent.parent / "src" / "code_rag" / "parsers" / "react_parser.py"
        content = react_parser.read_text(encoding='utf-8')

        # Check for encoding='utf-8' in subprocess calls
        subprocess_count = content.count('subprocess.run')
        encoding_count = content.count("encoding='utf-8'")

        # Should have 2 subprocess.run calls and 2 encodings
        assert subprocess_count == 2, f"Expected 2 subprocess.run, found {subprocess_count}"
        assert encoding_count >= 2, f"Expected at least 2 encoding='utf-8', found {encoding_count}"

        print(f"  [OK] Found {subprocess_count} subprocess.run calls with UTF-8 encoding\n")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}\n")
        return False


def test_node_modules_filtering():
    """Test 4: node_modules filtering"""
    print("Test 4: node_modules filtering")
    try:
        # Read build_and_index.py
        build_file = Path(__file__).parent.parent / "src" / "code_rag" / "graph" / "build_and_index.py"
        content = build_file.read_text(encoding='utf-8')

        # Check for node_modules filtering
        assert 'node_modules' in content, "Should check for node_modules"
        assert '.is_file()' in content, "Should check if path is file"

        print("  [OK] Filtering logic is present\n")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}\n")
        return False


def main():
    print("="*80)
    print("VERIFICATION OF ALL FIXES")
    print("="*80)
    print()

    tests = [
        test_full_name_property,
        test_react_parser_error_handling,
        test_subprocess_encoding,
        test_node_modules_filtering,
    ]

    results = [test() for test in tests]

    print("="*80)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("ALL FIXES VERIFIED SUCCESSFULLY!")
        print("="*80)
        return 0
    else:
        print("SOME TESTS FAILED - CHECK ABOVE")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
