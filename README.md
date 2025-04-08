# IBM CarbonSense powered by Watsonx

A comprehensive system for carbon footprint data analysis powered by IBM watsonx.ai. This system processes, analyzes, and provides insights into carbon footprint data from various sources.

For detailed technical documentation, please refer to:

- [Implementation Details](IMPLEMENTATION.md) - Technical implementation details and configurations
- [System Architecture](ARCHITECTURE.md) - System design and component interactions

## Prerequisites

### 1. System Requirements

- Python 3.8 or higher
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

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/IBM-CarbonSense-powered-by-Watsonx.git
cd IBM-CarbonSense-powered-by-Watsonx
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

This will install all required dependencies:
- python-dotenv (>=0.19.0)
- ibm-watsonx-ai (>=0.1.0)
- ibm-cos-sdk (>=2.0.0)
- pymilvus (>=2.3.0)
- python-docx (>=0.8.11)
- pydantic (>=1.8.2)
- pandas (>=2.0.0)
- openpyxl (>=3.0.0)
- ibm-watson (>=7.0.0)
- ibm-cloud-sdk-core (>=3.16.0)
- requests (>=2.31.0)
- numpy (>=1.24.0)
- tqdm (>=4.65.0)
- python-magic (>=0.4.27)
- python-magic-bin (>=0.4.14) [Windows only]

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
```

### 5. Fetch Milvus Certificates

Before using the system, you need to fetch the required Milvus certificates:

```powershell
# Fetch and install Milvus certificates
python -m carbonsense.main --mode fetch_certs
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
├── Data_RAW/              # Raw data files
│   └── industries/        # Industry data files
├── Data_processed/        # Processed data files
│   ├── Global/           # Global data
│   └── United States of America/  # US-specific data
├── Embeddings/           # Local embedding storage
│   ├── embeddings_30m/   # 30M model embeddings
│   ├── embeddings_125m/  # 125M model embeddings
│   └── embeddings_granite/# Granite model embeddings
├── milvus-grpc.crt       # Milvus certificate
├── src/                  # Source code
│   └── carbonsense/     # Main package
└── scripts/             # Utility scripts
```

## Usage

### 1. Generating Embeddings

#### Process All Files

```powershell
# Use default model (30M)
python -m carbonsense.main --mode generate

# Use specific model
python -m carbonsense.main --mode generate --model granite

# Process specific files
python -m carbonsense.main --mode generate --files Data_processed/Global/file1.xlsx Data_processed/Global/file2.xlsx
```

Available options:
- `--mode generate`: Required. Specifies the generation mode
- `--model`: Optional. Specify which model to use (30m, 125m, or granite)
- `--files`: Optional. List of specific files to process

### 2. Verification

#### Verify All Models

```powershell
python -m carbonsense.main --mode verify
```

#### Verify Specific Model

```powershell
python -m carbonsense.main --mode verify --model granite
```

Available options:
- `--mode verify`: Required. Specifies the verification mode
- `--model`: Optional. Specify which model to verify (30m, 125m, or granite)

### 3. Querying Data

#### Basic Query

```powershell
python -m carbonsense.main --mode rag_agent --query "What is the carbon footprint of 10 paper napkins?"
```

#### Query with Context

```powershell
python -m carbonsense.main --mode rag_agent --query "your question" --show_context
```

Available options:
- `--mode rag_agent`: Required. Specifies the query mode
- `--query`: Required. The question to ask about carbon footprint
- `--show_context`: Optional. Shows the sources used to generate the answer
- `--model`: Optional. Specify which model to use (30m, 125m, or granite)

### 4. Cleanup

#### Clean All Data

```powershell
python -m carbonsense.main --mode cleanup
```

Available options:
- `--mode cleanup`: Required. Specifies the cleanup mode

## Troubleshooting

### 1. Common Issues

#### API Authentication

- Verify API keys in .env file
- Check service URLs
- Ensure proper permissions

#### File Processing

- Check file formats
- Verify file permissions
- Ensure sufficient disk space

#### Embedding Generation

- Monitor rate limits
- Check model availability
- Verify network connectivity

#### Certificate Issues

- Ensure certificates are properly downloaded
- Verify certificate paths in .env file
- Check certificate permissions
- Update certificates if they expire

### 2. Error Messages

#### "API Key Invalid"

- Verify API key in .env file
- Check service status
- Ensure proper permissions

#### "File Not Found"

- Verify file path
- Check file permissions
- Ensure file exists

#### "Rate Limit Exceeded"

- Wait for cooldown period
- Reduce batch size
- Check service quotas

#### "Certificate Error"

- Run certificate fetch command
- Verify certificate paths
- Check certificate validity
- Update certificates if needed
