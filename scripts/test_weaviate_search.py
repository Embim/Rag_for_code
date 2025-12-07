"""Test Weaviate search directly."""
import sys
sys.path.insert(0, 'C:\\Users\\petrc\\Проекты\\Rag_for_Code')

from src.code_rag.graph.weaviate_indexer import WeaviateIndexer

def main():
    indexer = WeaviateIndexer(weaviate_url="http://localhost:8080")

    print("Testing search for 'equity instrument'...")
    results = indexer.search(query="equity instrument", limit=10)

    print(f"\nFound {len(results)} results:\n")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['name']} ({result['node_type']})")
        print(f"   File: {result['file_path']}")
        print(f"   Repo: {result['repository']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:100]}...")
        print()

    indexer.close()

if __name__ == "__main__":
    main()
