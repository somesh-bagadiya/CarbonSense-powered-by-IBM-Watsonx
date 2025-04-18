from setuptools import setup, find_packages

setup(
    name="carbonsense",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-dotenv>=0.19.0",
        "ibm-watsonx-ai>=0.1.0",
        "ibm-cos-sdk>=2.0.0",
        "pymilvus>=2.3.0",
        "python-docx>=0.8.11",
        "pydantic>=1.8.2",
        "pandas>=2.0.0",
        "openpyxl>=3.0.0",
        "ibm-watson>=7.0.0",
        "ibm-cloud-sdk-core>=3.16.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "python-magic>=0.4.27",
        "litellm>=1.30.0",  # Added for WatsonX integration with CrewAI
        "python-magic-bin>=0.4.14; sys_platform == 'win32'",
        "crewai>=0.28.0",  # CrewAI for multi-agent workflows
        "fastapi>=0.104.0",  # FastAPI framework for building APIs
        "uvicorn>=0.23.0",  # ASGI server for FastAPI
        "jinja2>=3.1.2",  # Templating engine for FastAPI
        "python-multipart>=0.0.6",  # For handling form data in FastAPI
    ],
    python_requires=">=3.11",
    author="IBM CarbonSense Team",
    description="A RAG-based system for carbon footprint analysis using IBM Watsonx",
    keywords="rag, watsonx, carbon-footprint, milvus, watson-discovery, crewai",
) 

# pip install -e .