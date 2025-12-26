"""
Index documents from SOP folder into Weaviate.

Usage:
    python scripts/index_documents.py --sop-dir "СОП" --batch-size 10
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.code_rag.parsers.document_parser import DocumentParser
from src.code_rag.graph.weaviate_indexer import WeaviateIndexer
from src.code_rag.graph.models import DocumentNode, create_node_id
from src.config import WeaviateConfig
from src.logger import get_logger

logger = get_logger(__name__)


def index_documents(
    sop_dir: Path,
    weaviate_config: WeaviateConfig,
    batch_size: int = 10,
    repository_name: str = "sop_documents"
):
    """
    Index all documents from SOP directory into Weaviate.

    Args:
        sop_dir: Path to SOP documents directory
        weaviate_config: Weaviate configuration
        batch_size: Number of documents to index in each batch
        repository_name: Virtual repository name for documents
    """
    logger.info(f"Indexing documents from: {sop_dir}")

    if not sop_dir.exists():
        logger.error(f"SOP directory not found: {sop_dir}")
        return

    # Initialize parser and indexer
    parser = DocumentParser()
    indexer = WeaviateIndexer(
        weaviate_url=weaviate_config.url,
        embedding_model=weaviate_config.embedding_model
    )

    # Ensure schema exists
    indexer.create_schema()

    # Find all .docx files
    docx_files = list(sop_dir.glob("**/*.docx"))
    logger.info(f"Found {len(docx_files)} .docx files")

    if not docx_files:
        logger.warning("No .docx files found in SOP directory")
        return

    # Parse and index documents
    document_nodes = []

    for doc_file in docx_files:
        logger.info(f"Parsing: {doc_file.name}")

        try:
            # Parse document
            parsed_doc = parser.parse_file(doc_file)

            if parsed_doc.errors:
                logger.warning(f"Parsing errors for {doc_file.name}: {parsed_doc.errors}")

            # Determine document type from filename or folder
            doc_type = _infer_document_type(doc_file)

            # Create relative path from SOP dir
            rel_path = doc_file.relative_to(sop_dir.parent)

            # Create DocumentNode
            node_id = create_node_id(
                repository=repository_name,
                file_path=str(rel_path)
            )

            doc_node = DocumentNode(
                id=node_id,
                name=parsed_doc.title,
                file_path=str(rel_path),
                document_type=doc_type,
                content=parsed_doc.full_text,
                author=parsed_doc.metadata.get('author', ''),
                created_date=parsed_doc.metadata.get('created', ''),
                modified_date=parsed_doc.metadata.get('modified', ''),
                sections_count=len(parsed_doc.sections),
                images_count=len(parsed_doc.images)
            )

            # Add metadata with image info
            doc_node.properties['images'] = [
                {
                    'filename': img['filename'],
                    'path': img['path'],
                    'width': img['width'],
                    'height': img['height']
                }
                for img in parsed_doc.images
            ]

            # Add section info
            doc_node.properties['sections'] = [
                {
                    'title': sec.title,
                    'level': sec.level,
                    'position': sec.position,
                    'has_images': len(sec.images) > 0,
                    'has_tables': len(sec.tables) > 0
                }
                for sec in parsed_doc.sections
            ]

            document_nodes.append(doc_node)

            logger.info(
                f"  Parsed {doc_file.name}: {len(parsed_doc.full_text)} chars, "
                f"{len(parsed_doc.sections)} sections, {len(parsed_doc.images)} images"
            )

        except Exception as e:
            logger.error(f"Failed to parse {doc_file.name}: {e}")
            continue

    # Index documents into Weaviate
    if document_nodes:
        logger.info(f"Indexing {len(document_nodes)} documents into Weaviate...")

        try:
            # Index using custom method for documents
            indexed_count = _index_documents_to_weaviate(
                indexer,
                document_nodes,
                batch_size
            )

            logger.info(f"Successfully indexed {indexed_count} documents")

            # Print summary
            print("\n" + "=" * 80)
            print("INDEXING SUMMARY")
            print("=" * 80)
            print(f"Total documents: {len(docx_files)}")
            print(f"Successfully indexed: {indexed_count}")
            print(f"Failed: {len(docx_files) - indexed_count}")
            print("=" * 80)

        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise

    else:
        logger.warning("No documents to index")


def _infer_document_type(file_path: Path) -> str:
    """Infer document type from filename or path."""
    name_lower = file_path.name.lower()

    if 'инструкц' in name_lower or 'instruction' in name_lower:
        return "Instruction"
    elif 'соп' in name_lower or 'sop' in name_lower:
        return "SOP"
    elif 'политик' in name_lower or 'policy' in name_lower:
        return "Policy"
    elif 'руководств' in name_lower or 'manual' in name_lower:
        return "Manual"
    elif 'процедур' in name_lower or 'procedure' in name_lower:
        return "Procedure"
    else:
        return "Document"


def _index_documents_to_weaviate(
    indexer: WeaviateIndexer,
    documents: list,
    batch_size: int
) -> int:
    """
    Index documents into Weaviate DocumentNode collection.

    Args:
        indexer: WeaviateIndexer instance
        documents: List of DocumentNode objects
        batch_size: Batch size for indexing

    Returns:
        Number of successfully indexed documents
    """
    indexed_count = 0

    try:
        collection = indexer.client.collections.get("DocumentNode")

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare batch data
            batch_contents = []
            batch_data = []

            for doc in batch:
                # Prepare searchable content
                content = f"Title: {doc.name}\n\n{doc.properties['content']}"
                batch_contents.append(content)

                # Prepare properties
                properties = {
                    "node_id": doc.id,
                    "node_type": doc.type.value,
                    "name": doc.name,
                    "file_path": doc.properties.get('file_path', ''),
                    "document_type": doc.properties.get('document_type', 'Document'),
                    "content": doc.properties.get('content', ''),
                    "author": doc.properties.get('author', ''),
                    "created_date": doc.properties.get('created_date', ''),
                    "modified_date": doc.properties.get('modified_date', ''),
                    "sections_count": doc.properties.get('sections_count', 0),
                    "images_count": doc.properties.get('images_count', 0),
                    "metadata": json.dumps(doc.properties)
                }

                batch_data.append(properties)

            # Generate embeddings
            logger.info(f"Generating embeddings for batch {i // batch_size + 1}...")
            embeddings = indexer.embedding_model.encode(
                batch_contents,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            # Insert into Weaviate
            logger.info(f"Inserting batch {i // batch_size + 1} into Weaviate...")

            with collection.batch.dynamic() as batch_inserter:
                for j, (props, vector) in enumerate(zip(batch_data, embeddings)):
                    try:
                        batch_inserter.add_object(
                            properties=props,
                            vector=vector.tolist()
                        )
                        indexed_count += 1
                    except Exception as e:
                        logger.error(f"Failed to insert document {props['name']}: {e}")

            logger.info(f"Batch {i // batch_size + 1} complete")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise

    return indexed_count


def main():
    parser = argparse.ArgumentParser(description="Index SOP documents into Weaviate")

    parser.add_argument(
        "--sop-dir",
        type=str,
        default="СОП",
        help="Path to SOP documents directory (default: СОП)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for indexing (default: 10)"
    )

    parser.add_argument(
        "--repository-name",
        type=str,
        default="sop_documents",
        help="Virtual repository name (default: sop_documents)"
    )

    args = parser.parse_args()

    # Load config
    from src.config import WeaviateConfig
    weaviate_config = WeaviateConfig.from_env()

    # Get SOP directory path
    sop_dir = Path(__file__).parent.parent / args.sop_dir

    # Index documents
    index_documents(
        sop_dir=sop_dir,
        weaviate_config=weaviate_config,
        batch_size=args.batch_size,
        repository_name=args.repository_name
    )


if __name__ == "__main__":
    main()
