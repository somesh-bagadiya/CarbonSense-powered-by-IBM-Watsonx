# CarbonSense powered by IBM Watsonx

A comprehensive system for carbon footprint data analysis powered by IBM watsonx.ai. This system processes, analyzes, and provides insights into carbon footprint data from various sources.

For detailed technical documentation, please refer to:

- [Implementation Details](IMPLEMENTATION.md) - Technical implementation details and configurations
- [System Architecture](ARCHITECTURE.md) - System design and component interactions

## Prerequisites

### 1. System Requirements

- Python 3.11 or higher
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space
- Internet connection for API access
- Windows operating system (for Windows-specific setup)

### 2. Required Services

- IBM Cloud Account
- Watsonx.ai access
- Milvus instance
- Cloud Object Storage (COS)
- Watson Discovery service
- Watson Speech-to-Text service (for voice queries)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/CarbonSense-powered-by-IBM-Watsonx.git
cd CarbonSense-powered-by-IBM-Watsonx
```

### 2. Create and Activate Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

### 3. Install Dependencies

```powershell
# Install the package and its dependencies
pip install -e .
```

This will install all required dependencies listed in setup.py, including:

- python-dotenv
- ibm-watsonx-ai
- ibm-cos-sdk
- pymilvus
- python-docx
- pandas, numpy, and tqdm
- crewai (for multi-agent workflows)
- fastapi and uvicorn (for the web interface)
- litellm (for WatsonX integration with CrewAI)

### 4. Environment Configuration

Create a `.env` file in the root directory with your credentials:

```env
# IBM Cloud Object Storage
COS_API_KEY=your_cos_api_key
COS_INSTANCE_ID=your_cos_instance_id
COS_ENDPOINT=your_cos_endpoint
BUCKET_NAME=your_bucket_name

# IBM Watsonx
WATSON_STUDIO_PROJECT_ID=your_project_id
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_URL=your_watsonx_url

# Milvus
MILVUS_GRPC_HOST=your_milvus_host
MILVUS_GRPC_PORT=your_milvus_port
MILVUS_CERT_PATH=path_to_milvus_cert

# Watson Discovery
WATSON_DISCOVERY_API_KEY=your_discovery_api_key
WATSON_DISCOVERY_URL=your_discovery_url
WATSON_DISCOVERY_PROJECT_ID=your_discovery_project_id

# Watson Speech-to-Text
IBM_STT_API_KEY=your_stt_api_key
IBM_STT_URL=your_stt_url
```

### 5. Fetch Milvus Certificates

Before using the system, you need to fetch the required Milvus certificates:

```powershell
# Fetch and install Milvus certificates
python -m src.carbonsense.main --mode fetch_certs
```

This command will:
- Create a backup of any existing certificate
- Fetch the new certificate from the Milvus server
- Save it in the root directory as `milvus-grpc.crt`
- Update the environment variables

## Directory Structure

The system creates the following structure:

```
.
├── Data_RAW/                # Raw data files
│   └── industries/          # Industry data files
├── Data_processed/          # Processed data files
│   ├── Global/              # Global data
│   └── United States of America/  # US-specific data
├── logs/                    # Logs for agent thoughts and processing steps
├── file_cache/              # Local file cache
├── Embeddings/              # Local embedding storage
│   ├── embeddings_30m/      # 30M model embeddings
│   ├── embeddings_125m/     # 125M model embeddings
│   └── embeddings_granite/  # Granite model embeddings
├── milvus-grpc.crt          # Milvus certificate
├── src/                     # Source code
│   └── carbonsense/         # Main package
│       ├── core/            # Core functionality
│       ├── services/        # Service integrations
│       ├── utils/           # Utility functions
│       ├── config/          # Configuration management
│       ├── web/             # Web interface
│       └── main.py          # Command-line interface
└── scripts/                 # Utility scripts
```

## Usage

### 1. Web Interface

Run the web server to access the dashboard interface:

```powershell
python -m src.carbonsense.web.run_server
```

Then open your browser to http://localhost:8000 to access the CarbonSense dashboard.

### 2. Generating Embeddings

#### Process All Files

```powershell
# Use default model (30M)
python -m src.carbonsense.main --mode generate

# Use specific model
python -m src.carbonsense.main --mode generate --model granite

# Process specific files
python -m src.carbonsense.main --mode generate --files Data_processed/Global/file1.xlsx Data_processed/Global/file2.xlsx
```

Available options:
- `--mode generate`: Required. Specifies the generation mode
- `--model`: Optional. Specify which model to use (30m, 125m, or granite)
- `--files`: Optional. List of specific files to process

### 3. Verification

```powershell
# Verify all model collections
python -m src.carbonsense.main --mode verify

# Verify specific model
python -m src.carbonsense.main --mode verify --model granite
```

### 4. Querying Data

#### Text Queries

```powershell
# Standard RAG-based agent
python -m src.carbonsense.main --mode rag_agent --query "What is the carbon footprint of 10 paper napkins?"

# Advanced CrewAI agent
python -m src.carbonsense.main --mode crew_agent --query "Compare the carbon footprint of paper vs plastic bags"
```

#### Voice Queries

```powershell
# Record for 15 seconds and process the voice query
python -m src.carbonsense.main --mode stt_query --record_duration 15
```

#### Additional Options

- `--mode`: Choose between `rag_agent` (standard), `crew_agent` (CrewAI), or `stt_query` (voice)
- `--query`: Your question about carbon footprint (required for text queries)
- `--show_context`: Shows the sources or agents used to generate the answer
- `--model`: Specify which model to use (30m, 125m, or granite)
- `--record_duration`: Duration of recording for voice input (seconds)
- `--debug`: Enable debug mode for detailed agent interactions
- `--no_cache`: Disable caching of query results
- `--sequential`: Use sequential process instead of hierarchical
- `--store_thoughts`: Store agent thoughts and reasoning in log files
- `--input_device`: Specify audio input device index for voice queries

### 5. System Maintenance

```powershell
# Clean up temporary files and caches
python -m src.carbonsense.main --mode cleanup
```

## Troubleshooting

### 1. Common Issues

#### API Authentication
- Verify API keys in .env file
- Check service URLs
- Ensure proper permissions

#### Audio Recording Issues
- Check microphone permissions
- Test with `--input_device` to select a specific microphone
- Verify Speech-to-Text credentials

#### Certificate Issues
- Run `fetch_certs` command to update certificates
- Verify certificate paths in .env file
- Check certificate permissions

### 2. Error Messages

#### "API Key Invalid"
- Verify API key in .env file
- Check service status
- Ensure proper permissions

#### "Rate Limit Exceeded"
- Wait for cooldown period
- Reduce batch size
- Check service quotas
