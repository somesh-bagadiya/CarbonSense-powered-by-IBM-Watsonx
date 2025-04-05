import logging
import argparse
from config.config_manager import ConfigManager
from core.embedding_generator import EmbeddingGenerator
from core.rag_generator import RAGGenerator

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='CarbonSense RAG Application')
    parser.add_argument('--mode', choices=['generate', 'query'], required=True,
                      help='Operation mode: generate embeddings or query the system')
    parser.add_argument('--query', help='Query to process in query mode')
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = ConfigManager()
        
        if args.mode == 'generate':
            logger.info("Starting embedding generation process")
            generator = EmbeddingGenerator(config)
            generator.process_all_files()
            logger.info("Embedding generation completed successfully")
            
        elif args.mode == 'query':
            if not args.query:
                logger.error("Query mode requires a query argument")
                return
                
            logger.info(f"Processing query: {args.query}")
            rag = RAGGenerator(config)
            answer = rag.generate_answer(args.query)
            print("\nAnswer:", answer)
            
            # Optionally show context
            context = rag.get_context(args.query)
            if context:
                print("\nRelevant context:")
                for chunk in context:
                    print(f"\nFrom {chunk['file_name']} (score: {chunk['score']:.2f}):")
                    print(chunk['text'])
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 