import argparse
import logging
from .core.embedding_generator import EmbeddingGenerator
from .config.config_manager import ConfigManager
from .utils.logger import setup_logger
import os
from .services.milvus_service import MilvusService
import sys
from .core.carbon_agent import CarbonAgent
import requests
from pathlib import Path
import ssl
import socket
import shutil
import time

# Set up logger
logger = setup_logger(__name__)

def fetch_certificates(config: ConfigManager) -> None:
    """Fetch and save Milvus certificates."""
    try:
        milvus_config = config.get_milvus_config()
        host = milvus_config.get('host')
        port = int(milvus_config.get('port', 30902))  # default port if missing
        
        # Use root directory for certificate
        cert_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        cert_file = os.path.join(cert_path, "milvus-grpc.crt")
        
        # Backup existing certificate if it exists
        if os.path.exists(cert_file):
            backup_file = os.path.join(cert_path, f"milvus-grpc_backup_{int(time.time())}.crt")
            shutil.copy2(cert_file, backup_file)
            logger.info(f"✅ Existing certificate backed up to: {backup_file}")
        
        logger.info(f"Fetching certificate from {host}:{port}...")
        
        try:
            # Attempt to fetch server certificate
            cert = ssl.get_server_certificate((host, port))
            
            # Write the certificate to a file
            with open(cert_file, "w") as f:
                f.write(cert)
            
            logger.info(f"✅ Certificate successfully saved to: {cert_file}")
            
            # Update environment variable
            os.environ['MILVUS_CERT_PATH'] = cert_path
            logger.info(f"Certificate path updated to {cert_path}")
            
        except socket.gaierror as e:
            logger.error(f"❌ DNS resolution failed for host: {host}")
            logger.error(f"Error: {e}")
            raise
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch certificate from {host}:{port}")
            logger.error(f"Error: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in fetch_certificates: {str(e)}")
        raise

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="CarbonSense - Carbon Footprint Analysis")
    parser.add_argument("--mode", choices=["generate", "rag_agent", "verify", "cleanup", "fetch_certs"], required=True, help="Operation mode")
    parser.add_argument("--model", choices=["125m", "granite", "30m"], help="Model to use for embeddings")
    parser.add_argument("--query", help="Query string for query mode")
    parser.add_argument("--show_context", action="store_true", help="Show context in query results")
    parser.add_argument("--files", nargs="+", help="Specific files to process")
    args = parser.parse_args()
    
    try:
        # Initialize config manager
        config = ConfigManager()
        
        if args.mode == "fetch_certs":
            fetch_certificates(config)
            return
            
        if args.mode == "generate":
            generator = EmbeddingGenerator(config)
            # Set model flags based on argument
            use_125m = args.model == "125m"
            use_granite = args.model == "granite"
            generator.process_all_files(use_125m, use_granite, args.files)
            
        elif args.mode == "rag_agent":
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
            
            # Define all model collections
            collections = {
                "30m": "carbon_embeddings_30m",
                "125m": "carbon_embeddings_125m",
                "granite": "carbon_embeddings_granite"
            }
            
            # If model is specified, only verify that model
            if args.model:
                if args.model not in collections:
                    logger.error(f"Invalid model: {args.model}. Must be one of: {', '.join(collections.keys())}")
                    return
                collections = {args.model: collections[args.model]}
            
            for model_name, collection_name in collections.items():
                logger.info(f" Verifying {model_name} model collection:")
                logger.info("=" * 80)
                
                stats = milvus.verify_collection(collection_name)
                
                if "error" in stats:
                    logger.error(f" Error: {stats['error']}")
                else:
                    logger.info(f" Total vectors: {stats['num_entities']}")
                    logger.info(f" Unique files: {len(stats['unique_files'])}")
                    logger.info(f" Model types: {', '.join(stats['model_types'])}")
                    
                    logger.info(" Data Consistency:")
                    logger.info("-" * 40)
                    logger.info(f" Has valid embeddings: {stats.get('has_valid_embeddings', 'Unknown')}")
                    logger.info(f" Has valid metadata: {stats.get('has_valid_metadata', 'Unknown')}")
                    if 'data_consistency_error' in stats:
                        logger.error(f" Data consistency error: {stats['data_consistency_error']}")
                    
                    logger.info(" Files in collection:")
                    for file in stats['unique_files']:
                        print(f"- {file}")
                    
                    logger.info(" Schema:")
                    # Convert schema to string representation
                    schema_str = f"Description: {stats['schema'].description}\nFields:\n"
                    for field in stats['schema'].fields:
                        schema_str += f"- {field.name}: {field.dtype}\n"
                    logger.info(schema_str)
                    
                    logger.info(" Indexes:")
                    # Convert indexes to string representation
                    indexes_str = ""
                    for idx in stats['indexes']:
                        indexes_str += f"- {idx.field_name}: {idx.params}\n"
                    logger.info(indexes_str)
                
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