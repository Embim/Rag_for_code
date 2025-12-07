"""
Complete pipeline: Repository → Parsing → Graph → Weaviate

This script demonstrates the full workflow:
1. Load repository
2. Parse code files
3. Build Neo4j knowledge graph
4. Index nodes in Weaviate
5. Link frontend-backend APIs
"""

from pathlib import Path
from typing import Optional, List
import argparse

from ..repo_loader import RepositoryLoader
from ..parsers import get_parser
from .neo4j_client import Neo4jClient
from .graph_builder import GraphBuilder
from .api_linker import APILinker
from .weaviate_indexer import WeaviateIndexer
from .models import ComponentNode, EndpointNode
from ...logger import get_logger


logger = get_logger(__name__)


class GraphPipeline:
    """
    Complete pipeline for building and indexing knowledge graph.

    Orchestrates the entire process from repository to searchable graph.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        weaviate_url: str = "http://localhost:8080",
        embedding_model: str = "BAAI/bge-m3"
    ):
        """
        Initialize pipeline components.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            weaviate_url: Weaviate connection URL
            embedding_model: Embedding model for Weaviate
        """
        # Initialize components
        self.neo4j_client = Neo4jClient(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password
        )

        self.graph_builder = GraphBuilder(self.neo4j_client)
        self.api_linker = APILinker(use_openapi=True)

        self.weaviate_indexer = WeaviateIndexer(
            weaviate_url=weaviate_url,
            embedding_model=embedding_model,
            neo4j_client=self.neo4j_client
        )

        self.repo_loader = RepositoryLoader()

    def run(
        self,
        source: str,
        name: Optional[str] = None,
        branch: str = "main",
        clear_existing: bool = False,
        link_apis: bool = True,
        index_weaviate: bool = True
    ) -> dict:
        """
        Run complete pipeline.

        Args:
            source: Repository URL or local path
            name: Optional repository name
            branch: Git branch to use
            clear_existing: Whether to clear existing data
            link_apis: Whether to run API linking
            index_weaviate: Whether to index in Weaviate

        Returns:
            Statistics dictionary
        """
        stats = {
            'repository': name or source,
            'files_parsed': 0,
            'entities_found': 0,
            'nodes_created': 0,
            'relationships_created': 0,
            'api_links_created': 0,
            'nodes_indexed': 0
        }

        logger.info(f"Starting pipeline for: {source}")

        # Step 1: Load repository
        logger.info("Step 1: Loading repository...")
        repo_info = self.repo_loader.load(
            source=source,
            name=name,
            branch=branch
        )
        logger.info(f"Repository loaded: {repo_info.name}")
        logger.info(f"  Type: {repo_info.project_type}")
        logger.info(f"  Languages: {', '.join(repo_info.languages or [])}")
        logger.info(f"  Frameworks: {', '.join(repo_info.frameworks or [])}")

        # Step 2: Parse code files
        logger.info("\nStep 2: Parsing code files...")
        parse_results = []

        code_files = list(repo_info.path.rglob('*.py')) + \
                     list(repo_info.path.rglob('*.js')) + \
                     list(repo_info.path.rglob('*.jsx')) + \
                     list(repo_info.path.rglob('*.ts')) + \
                     list(repo_info.path.rglob('*.tsx'))

        for file_path in code_files:
            # Skip if not a file (e.g., directories with extensions in name)
            if not file_path.is_file():
                continue

            # Skip test files and migrations
            if 'test' in file_path.name.lower() or 'migration' in str(file_path):
                continue

            # Skip node_modules and other vendor directories
            path_str = str(file_path)
            if any(skip in path_str for skip in ['node_modules', 'venv', '.venv', 'dist', 'build', '__pycache__']):
                continue

            parser = get_parser(file_path)
            if parser:
                try:
                    result = parser.parse_file(file_path)
                    parse_results.append((file_path, result))

                    stats['files_parsed'] += 1
                    stats['entities_found'] += len(result.entities)

                    logger.debug(f"Parsed {file_path.name}: {len(result.entities)} entities")

                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")

        logger.info(f"Parsed {stats['files_parsed']} files, found {stats['entities_found']} entities")

        # Step 3: Build knowledge graph
        logger.info("\nStep 3: Building knowledge graph...")

        if clear_existing:
            logger.warning("Clearing existing graph data...")
            self.neo4j_client.clear_database()

        graph_stats = self.graph_builder.build_graph(repo_info, parse_results)

        stats['nodes_created'] = graph_stats['nodes_created']
        stats['relationships_created'] = graph_stats['relationships_created']

        logger.info(f"Graph built: {stats['nodes_created']} nodes, {stats['relationships_created']} relationships")

        # Step 4: Link frontend-backend APIs (optional)
        if link_apis:
            logger.info("\nStep 4: Linking frontend-backend APIs...")

            try:
                # Get component nodes (frontend)
                component_dicts = self.neo4j_client.find_nodes(
                    node_type=self.graph_builder.nodes.get(ComponentNode),
                    limit=10000
                )

                # Get endpoint nodes (backend)
                endpoint_dicts = self.neo4j_client.find_nodes(
                    node_type=self.graph_builder.nodes.get(EndpointNode),
                    limit=10000
                )

                # Convert to proper node types
                component_nodes = [
                    ComponentNode(
                        id=n['id'],
                        name=n['name'],
                        props_type=n.get('props_type'),
                        hooks_used=n.get('hooks_used', '').split(',') if n.get('hooks_used') else [],
                        properties=n
                    )
                    for n in component_dicts
                ]

                endpoint_nodes = [
                    EndpointNode(
                        id=n['id'],
                        name=n['name'],
                        http_method=n.get('http_method', 'GET'),
                        path=n.get('path', '/'),
                        properties=n
                    )
                    for n in endpoint_dicts
                ]

                # Link APIs
                api_links = self.api_linker.link_api_calls(component_nodes, endpoint_nodes)

                # Save relationships
                if api_links:
                    links_created = self.neo4j_client.create_relationships_batch(api_links)
                    stats['api_links_created'] = links_created
                    logger.info(f"Created {links_created} API links")

                # Find orphaned calls (for debugging)
                orphaned = self.api_linker.find_orphaned_api_calls(component_nodes, endpoint_nodes)
                if orphaned:
                    logger.warning(f"Found {len(orphaned)} orphaned API calls")
                    for call in orphaned[:5]:  # Show first 5
                        logger.warning(f"  - {call['component']}: {call['method']} {call['url']}")

            except Exception as e:
                logger.error(f"API linking failed: {e}")

        # Step 5: Index in Weaviate (optional)
        if index_weaviate:
            logger.info("\nStep 5: Indexing in Weaviate...")

            try:
                # Create schema
                self.weaviate_indexer.create_schema()

                # Index all nodes from Neo4j
                indexed = self.weaviate_indexer.index_from_neo4j(batch_size=50)
                stats['nodes_indexed'] = indexed

                logger.info(f"Indexed {indexed} nodes in Weaviate")

            except Exception as e:
                logger.error(f"Weaviate indexing failed: {e}")

        # Step 6: Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Repository: {stats['repository']}")
        logger.info(f"Files parsed: {stats['files_parsed']}")
        logger.info(f"Entities found: {stats['entities_found']}")
        logger.info(f"Nodes created: {stats['nodes_created']}")
        logger.info(f"Relationships created: {stats['relationships_created']}")
        logger.info(f"API links created: {stats['api_links_created']}")
        logger.info(f"Nodes indexed: {stats['nodes_indexed']}")
        logger.info("=" * 60)

        return stats

    def close(self):
        """Close all connections."""
        self.neo4j_client.close()
        self.weaviate_indexer.close()


def build_and_index(
    repos_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    weaviate_url: str = "http://localhost:8080",
    embedding_model: str = "BAAI/bge-m3",
    clear_existing: bool = False,
    link_apis: bool = True,
    index_weaviate: bool = True
) -> dict:
    """
    Convenience function to build and index a repository.

    Args:
        repos_dir: Repository path or URL
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        weaviate_url: Weaviate connection URL
        embedding_model: Embedding model for Weaviate
        clear_existing: Whether to clear existing data
        link_apis: Whether to run API linking
        index_weaviate: Whether to index in Weaviate

    Returns:
        Statistics dictionary
    """
    pipeline = GraphPipeline(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        weaviate_url=weaviate_url,
        embedding_model=embedding_model
    )

    try:
        stats = pipeline.run(
            source=repos_dir,
            clear_existing=clear_existing,
            link_apis=link_apis,
            index_weaviate=index_weaviate
        )
        return stats
    finally:
        pipeline.close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build and index knowledge graph from code repository"
    )

    parser.add_argument(
        'source',
        help='Repository URL or local path'
    )

    parser.add_argument(
        '--name',
        help='Repository name (optional, inferred from URL/path if not provided)'
    )

    parser.add_argument(
        '--branch',
        default='main',
        help='Git branch to use (default: main)'
    )

    parser.add_argument(
        '--neo4j-uri',
        default='bolt://localhost:7687',
        help='Neo4j connection URI'
    )

    parser.add_argument(
        '--neo4j-user',
        default='neo4j',
        help='Neo4j username'
    )

    parser.add_argument(
        '--neo4j-password',
        default='password',
        help='Neo4j password'
    )

    parser.add_argument(
        '--weaviate-url',
        default='http://localhost:8080',
        help='Weaviate connection URL'
    )

    parser.add_argument(
        '--embedding-model',
        default='BAAI/bge-m3',
        help='Embedding model for Weaviate'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing graph data before building'
    )

    parser.add_argument(
        '--no-api-linking',
        action='store_true',
        help='Skip API linking step'
    )

    parser.add_argument(
        '--no-weaviate',
        action='store_true',
        help='Skip Weaviate indexing step'
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = GraphPipeline(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        weaviate_url=args.weaviate_url,
        embedding_model=args.embedding_model
    )

    try:
        # Run pipeline
        stats = pipeline.run(
            source=args.source,
            name=args.name,
            branch=args.branch,
            clear_existing=args.clear,
            link_apis=not args.no_api_linking,
            index_weaviate=not args.no_weaviate
        )

        # Exit successfully
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

    finally:
        pipeline.close()


if __name__ == '__main__':
    exit(main())
