# Setup script for CarbonSense on Windows
Write-Host "Setting up CarbonSense environment..."

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate

# Install dependencies
Write-Host "Installing dependencies..."
pip install -e ".[dev]"

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file template..."
    @"
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
"@ | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host "Please update the .env file with your credentials"
}

# Create data directories if they don't exist
Write-Host "Creating data directories..."
$directories = @(
    "Data_RAW",
    "Data_processed/Global",
    "Data_processed/United States of America"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force
    }
}

Write-Host "`nSetup complete! Next steps:"
Write-Host "1. Update the .env file with your credentials"
Write-Host "2. Run 'python -m carbonsense.main --mode generate' to generate embeddings"
Write-Host "3. Run 'uvicorn carbonsense.api.app:app --reload' to start the API server" 