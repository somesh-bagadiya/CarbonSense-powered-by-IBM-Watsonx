# IBM CarbonSense powered by Watsonx

A comprehensive system for carbon footprint data analysis powered by IBM watsonx.ai. This system processes, analyzes, and provides insights into carbon footprint data from various sources.

For detailed technical documentation, please refer to:

- [Implementation Details](IMPLEMENTATION.md) - Technical implementation details and configurations
- [System Architecture](ARCHITECTURE.md) - System design and component interactions

## Prerequisites

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

### 2. Setup Environment (Windows)

```powershell
# Run the setup script
.\scripts\setup.ps1
```

The setup script will:

- Create a virtual environment
- Install dependencies
- Create a `.env` file template
- Set up required directories

### 3. Environment Configuration

Update the `.env` file with your credentials:

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

### 4. Fetch Milvus Certificates

Before using the system, you need to fetch the required Milvus certificates:

```bash
# Fetch and install Milvus certificates
python -m carbonsense.main --mode fetch_certs
```

This command will:

- Create a `milvus_cert` directory
- Download the root CA certificate
- Download the client certificate
- Update the environment variables

The certificates will be stored in the `milvus_cert` directory by default. You can specify a different path in your `.env` file using the `MILVUS_CERT_PATH` variable.

## Directory Structure

The setup script creates the following structure:

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
├── milvus_cert/          # Milvus certificates
│   ├── root_ca.pem       # Root CA certificate
│   └── client.pem        # Client certificate
├── src/                  # Source code
│   └── carbonsense/     # Main package
└── scripts/             # Utility scripts
```

## Usage

### 1. Generating Embeddings

#### Process All Files

```bash
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

```bash
python -m carbonsense.main --mode verify
```

#### Verify Specific Model

```bash
python -m carbonsense.main --mode verify --model granite
```

Available options:

- `--mode verify`: Required. Specifies the verification mode
- `--model`: Optional. Specify which model to verify (30m, 125m, or granite)

The verification process will:

- Check collection statistics
- Verify data consistency
- List all files in the collection
- Display schema and index information
- Show model types and unique files

### 3. Querying Data

#### Basic Query

```bash
python -m carbonsense.main --mode rag_agent --query "What is the carbon footprint of 10 paper napkins?"
```

#### Query with Context

```bash
python -m carbonsense.main --mode rag_agent --query "your question" --show_context
```

Available options:

- `--mode rag_agent`: Required. Specifies the query mode
- `--query`: Required. The question to ask about carbon footprint
- `--show_context`: Optional. Shows the sources used to generate the answer
- `--model`: Optional. Specify which model to use (30m, 125m, or granite)

### 4. Cleanup

#### Clean All Data

```bash
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
