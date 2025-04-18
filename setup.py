from setuptools import setup, find_packages

setup(
    name="carbonsense",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Core dependencies
        "python-dotenv==1.1.0",
        "ibm-watsonx-ai==1.3.6",
        "ibm-cos-sdk==2.14.0",
        "pymilvus==2.5.6",
        "pydantic==2.11.3",
        "ibm-watson==9.0.0",
        "ibm-cloud-sdk-core==3.23.0",
        "requests==2.32.2",
        "litellm==1.60.2",
        "crewai==0.114.0",
        "crewai-tools==0.40.1",
        
        # Data processing
        "pandas==2.2.3",
        "numpy==2.2.4",
        "scipy==1.15.2",
        "openpyxl==3.1.5",
        "python-docx==1.1.2",
        "pypdf==5.4.0",
        
        # Audio processing
        "sounddevice==0.5.1",
        
        # Utilities
        "tqdm==4.67.1",
        "python-magic==0.4.27",
        "python-magic-bin==0.4.14; sys_platform == 'win32'",
        
        # Web interface
        "fastapi==0.115.9",
        "uvicorn==0.34.0",
        "jinja2==3.1.6",
        "python-multipart==0.0.20",
        "aiofiles==24.1.0",
        "sse-starlette==2.2.1",
        
        # Concurrent processing
        "aiohttp==3.11.16",
        "httpx==0.27.2",
        "anyio==4.9.0",
        "asyncio>=3.4.3",
        
        # Additional dependencies
        "pyright==1.1.399",  # For code analysis
        "pyarrow==19.0.1",  # For efficient data serialization
        "pillow==11.1.0",  # For image processing
        "typer==0.15.2",  # For CLI interfaces
        "pywin32==310; sys_platform == 'win32'",  # Windows-specific utilities
        "rich==13.9.4",  # Enhanced terminal output
    ],
    python_requires=">=3.11",
    author="IBM CarbonSense Team",
    description="A RAG-based system for carbon footprint analysis using IBM Watsonx",
    keywords="rag, watsonx, carbon-footprint, milvus, watson-discovery, crewai",
) 

# pip install -e .