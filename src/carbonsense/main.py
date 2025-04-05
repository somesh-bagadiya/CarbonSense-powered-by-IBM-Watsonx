import argparse
import logging
from .core.embedding_generator import EmbeddingGenerator
from .core.query_service import QueryService
from .config.config_manager import ConfigManager
from .utils.logger import setup_logger
import os
from .services.milvus_service import MilvusService
import sys

# Set up logger
logger = setup_logger(__name__)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="CarbonSense CLI")
    parser.add_argument("--mode", choices=["generate", "query", "verify", "cleanup"], required=True, help="Operation mode")
    parser.add_argument("--model", choices=["125m", "granite", "30m"], help="Model to use for embeddings")
    parser.add_argument("--query", help="Query text for RAG")
    parser.add_argument("--show_context", action="store_true", help="Show context in query results")
    parser.add_argument("--files", nargs="+", help="Specific files to process")
    args = parser.parse_args()
    
    try:
        # Initialize config manager
        config = ConfigManager()
        
        if args.mode == "generate":
            generator = EmbeddingGenerator(config)
            # Set model flags based on argument
            use_125m = args.model == "125m"
            use_granite = args.model == "granite"
            generator.process_all_files(use_125m, use_granite, args.files)
            
        elif args.mode == "query":
            query_service = QueryService(config)
            result = query_service.process_query(args.query, args.show_context)
            
            # Print results
            print("\nQuery Results:")
            print("=" * 80)
            print(f"Query: {result['query']}\n")
            
            for i, r in enumerate(result['results'], 1):
                print(f"\nResult {i}:")
                print(f"File: {r['file_name']}")
                print(f"Score: {r['score']:.4f}")
                print(f"Text: {r['chunk']}")
                if 'context' in r:
                    print(f"Context: {r['context']}")
                print("-" * 80)
                
        elif args.mode == "verify":
            milvus = MilvusService(config)
            stats = milvus.verify_collection("carbon_embeddings_granite")
            
            if "error" in stats:
                print(f"Error: {stats['error']}")
            else:
                print("\nCollection Verification Results:")
                print("=" * 80)
                print(f"Total vectors: {stats['num_entities']}")
                print(f"Unique files: {len(stats['unique_files'])}")
                print(f"Model types: {', '.join(stats['model_types'])}")
                
                print("\nData Consistency:")
                print("-" * 40)
                print(f"Has valid embeddings: {stats.get('has_valid_embeddings', 'Unknown')}")
                print(f"Has valid metadata: {stats.get('has_valid_metadata', 'Unknown')}")
                if 'data_consistency_error' in stats:
                    print(f"Data consistency error: {stats['data_consistency_error']}")
                
                print("\nFiles in collection:")
                for file in stats['unique_files']:
                    print(f"- {file}")
                
                print("\nSchema:")
                print(stats['schema'])
                print("\nIndexes:")
                print(stats['indexes'])
                
        elif args.mode == "cleanup":
            try:
                from cleanup import CleanupManager
                cleanup_manager = CleanupManager()
                
                print("\nStarting cleanup process...")
                print("=" * 80)
                
                # Clean up COS bucket
                print("\nCleaning up COS bucket...")
                cleanup_manager.cleanup_cos_bucket()
                
                # Clean up Milvus collections
                print("\nCleaning up Milvus collections...")
                cleanup_manager.cleanup_milvus()
                
                # Clean up local embeddings
                print("\nCleaning up local embeddings...")
                cleanup_manager.cleanup_local_embeddings()
                
                print("\nCleanup completed successfully")
                
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 