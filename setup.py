from setuptools import setup, find_packages

setup(
    name="carbonsense",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-dotenv>=0.19.0",
        "ibm-watsonx-ai>=0.1.0",
        "ibm-boto3>=2.0.0",
        "pymilvus>=2.3.0",
        "python-docx>=0.8.11",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.2",
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "isort>=5.9.3",
        "flake8>=3.9.2",
        "mypy>=0.910",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.9.3",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ]
    },
    python_requires=">=3.8",
    author="IBM CarbonSense Team",
    description="A RAG-based system for carbon footprint analysis using IBM Watsonx",
    keywords="rag, watsonx, carbon-footprint, milvus",
) 