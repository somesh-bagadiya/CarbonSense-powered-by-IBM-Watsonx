import logging
import sys
import subprocess

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('embedding_generation.log')
        ]
    )

def run_command(model: str):
    """Run the embedding generation command for a specific model."""
    try:
        cmd = f"python -m carbonsense.main --mode generate --model {model}"
        logging.info(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        logging.info(result.stdout)
        if result.stderr:
            logging.warning(result.stderr)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running command for model {model}: {str(e)}")
        logging.error(f"Command output: {e.stdout}")
        logging.error(f"Command error: {e.stderr}")
        raise

def main():
    """Main function to generate embeddings for all files."""
    try:
        # Setup logging
        setup_logging()
        
        # Process files with each model type
        models = ["30m", "125m", "granite"]
        
        for model in models:
            logging.info(f"\nProcessing files with {model} model...")
            run_command(model)
        
        logging.info("\nEmbedding generation completed successfully!")
        
    except Exception as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise

if __name__ == "__main__":
    main() 