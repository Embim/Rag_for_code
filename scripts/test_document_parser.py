"""
Test document parser on SOP document.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_rag.parsers.document_parser import DocumentParser


def main():
    # Path to SOP document
    doc_path = Path(__file__).parent.parent / "СОП" / "Инструкция по корп.docx"

    if not doc_path.exists():
        print(f"Document not found: {doc_path}")
        return

    print(f"Parsing document: {doc_path}")
    print("=" * 80)

    # Create parser
    parser = DocumentParser()

    # Parse document
    result = parser.parse_file(doc_path)

    # Print results
    print(f"\nTitle: {result.title}")
    print(f"Sections: {len(result.sections)}")
    print(f"Images: {len(result.images)}")
    print(f"Full text length: {len(result.full_text)} chars")

    if result.metadata:
        print("\nMetadata:")
        for key, value in result.metadata.items():
            if value:
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("SECTIONS:")
    print("=" * 80)

    for i, section in enumerate(result.sections[:5]):  # First 5 sections
        print(f"\n{i+1}. {section.title or '(No title)'}")
        print(f"   Level: {section.level}, Position: {section.position}")
        print(f"   Images: {len(section.images)}, Tables: {len(section.tables)}")
        print(f"   Content preview: {section.content[:200]}..." if len(section.content) > 200 else f"   Content: {section.content}")

    if len(result.sections) > 5:
        print(f"\n... and {len(result.sections) - 5} more sections")

    print("\n" + "=" * 80)
    print("IMAGES:")
    print("=" * 80)

    for i, img in enumerate(result.images[:5]):  # First 5 images
        print(f"\n{i+1}. {img['filename']}")
        print(f"   Size: {img['width']}x{img['height']}, Format: {img['format']}")
        print(f"   Path: {img['path']}")

    if len(result.images) > 5:
        print(f"\n... and {len(result.images) - 5} more images")

    if result.errors:
        print("\n" + "=" * 80)
        print("ERRORS:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
