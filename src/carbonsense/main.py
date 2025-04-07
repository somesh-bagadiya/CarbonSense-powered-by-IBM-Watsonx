import argparse
import logging
from .core.embedding_generator import EmbeddingGenerator
from .core.query_service import QueryService
from .config.config_manager import ConfigManager
from .utils.logger import setup_logger
import os
from .services.milvus_service import MilvusService
import sys
# from ..core.carbon_agent import CarbonAgent

# Set up logger
logger = setup_logger(__name__)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="CarbonSense - Carbon Footprint Analysis")
    parser.add_argument("--mode", choices=["generate", "query", "verify", "cleanup"], required=True, help="Operation mode")
    parser.add_argument("--model", choices=["125m", "granite", "30m"], help="Model to use for embeddings")
    parser.add_argument("--query", help="Query string for query mode")
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
            if not args.query:
                print("Error: --query argument is required for query mode")
                return
            
            try:
                agent = CarbonAgent(config)
                result = agent.process_query(args.query)
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                    return
                
                print("\nResponse:")
                print("=" * 80)
                print(result["response"])
                
                if args.show_context:
                    print("\nSources:")
                    print("-" * 40)
                    for source in result["sources"]:
                        print(f"- {source}")
                
                print(f"\nConfidence Score: {result['confidence']:.2f}")
                
            except Exception as e:
                print(f"Error processing query: {str(e)}")
            
        elif args.mode == "verify":
            milvus = MilvusService(config)
            
            # Check all model collections
            collections = {
                "30m": "carbon_embeddings_30m",
                "125m": "carbon_embeddings_125m",
                "granite": "carbon_embeddings_granite"
            }
            
            for model_name, collection_name in collections.items():
                print(f"\nVerifying {model_name} model collection:")
                print("=" * 80)
                
                stats = milvus.verify_collection(collection_name)
                
                if "error" in stats:
                    print(f"Error: {stats['error']}")
                else:
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
                from .services.cleanup_service import CleanupService
                cleanup_service = CleanupService(config)
                
                print("\nStarting cleanup process...")
                print("=" * 80)
                
                # Use the cleanup_all method which handles everything properly
                cleanup_service.cleanup_all()
                
                print("\nCleanup completed successfully")
                
            except Exception as e:
                logger.error(f"Cleanup failed: {str(e)}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 