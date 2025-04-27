#!/usr/bin/env python3
"""
Run script for the CarbonSense Web Dashboard
This starts the FastAPI server with uvicorn
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

def main():
    """Run the FastAPI server with Uvicorn"""
    print("Starting CarbonSense Web Dashboard...")
    print(f"Project root: {project_root}")
    
    # Set environment variables for development
    os.environ["CARBONSENSE_ENV"] = "development"
    
    # Configure and run uvicorn server
    uvicorn.run(
        "src.carbonsense.web.app:app",  # Use the full module path
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True,
        workers=1  # Use single worker for development
    )

if __name__ == "__main__":
    main() 