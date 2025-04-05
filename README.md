# CarbonSense - Powered by IBM Watsonx

A RAG-based system for carbon footprint analysis using IBM Watsonx, Milvus, and IBM Cloud Object Storage.

## Architecture

The system consists of three main components:

1. **Frontend/UI**: A FastAPI-based API that handles user interactions
2. **RAG Script**: Core RAG implementation for question answering
3. **Embedding Generation**: Process for generating and storing document embeddings

## Prerequisites

- Python 3.8 or higher
- IBM Cloud account with:
  - Watsonx.ai access
  - Cloud Object Storage bucket
  - Milvus database instance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/IBM-CarbonSense-powered-by-Watsonx.git
cd IBM-CarbonSense-powered-by-Watsonx
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
COS_API_KEY=your_cos_api_key
COS_INSTANCE_ID=your_cos_instance_id
COS_ENDPOINT=your_cos_endpoint
BUCKET_NAME=your_bucket_name
WATSON_STUDIO_PROJECT_ID=your_project_id
MILVUS_GRPC_HOST=your_milvus_host
MILVUS_GRPC_PORT=your_milvus_port
MILVUS_CERT_PATH=path_to_milvus_cert
```

## Usage

### Starting the API Server

```bash
uvicorn carbonsense.api.app:app --reload
```

The API will be available at `http://localhost:8000`

### Generating Embeddings

```bash
python -m carbonsense.main --mode generate
```

### Querying the System

```bash
python -m carbonsense.main --mode query --query "What is carbon footprint?"
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

1. Format code:
```bash
black src/
```

2. Sort imports:
```bash
isort src/
```

3. Type checking:
```bash
mypy src/
```

4. Linting:
```bash
flake8 src/
```

## Project Structure

```
src/carbonsense/
├── api/            # FastAPI application
├── config/         # Configuration management
├── core/           # Core RAG and embedding logic
├── models/         # Data models and schemas
├── services/       # External service integrations
├── utils/          # Utility functions
├── tests/          # Test suite
│   ├── unit/      # Unit tests
│   └── integration/# Integration tests
├── docs/          # Documentation
└── scripts/       # Utility scripts
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
