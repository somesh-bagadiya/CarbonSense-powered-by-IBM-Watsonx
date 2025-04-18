import uvicorn
import os
import sys
import logging
from pathlib import Path

# Configure logging with less verbosity
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Simplified format
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add the project root to the Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

print("\n" + "="*50)
print("Starting CarbonSense Web Dashboard")
print(f"Server address: http://localhost:8000")
print("="*50 + "\n")

if __name__ == "__main__":
    # Run the FastAPI server with reduced logging
    uvicorn.run(
        "src.carbonsense.web.app:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"  # Changed from debug to info
    ) 