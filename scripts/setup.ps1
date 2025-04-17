# Setup script for CarbonSense on Windows
Write-Host "Setting up CarbonSense environment..."

# Check Python version
Write-Host "Checking Python version..."
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$versionParts = $pythonVersion.Split(".")
$major = [int]$versionParts[0]
$minor = [int]$versionParts[1]

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
    Write-Host "Error: Python 3.11 or higher is required. Current version: $pythonVersion" -ForegroundColor Red
    Write-Host "Please install Python 3.11+ from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
Write-Host "Python version $pythonVersion detected - meets requirements (>=3.11)" -ForegroundColor Green

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

# Verify BeeAI framework installation
Write-Host "Verifying BeeAI framework installation..." -ForegroundColor Cyan
try {
    python -c "import beeai_framework; print(f'BeeAI framework {beeai_framework.__version__} successfully installed')"
    Write-Host "BeeAI framework installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Installing BeeAI framework explicitly..." -ForegroundColor Yellow
    pip install beeai-framework>=0.1.14
    Write-Host "BeeAI framework installation completed." -ForegroundColor Green
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file template..."
    @"
# IBM Cloud Object Storage
COS_API_KEY=your_cos_api_key
COS_INSTANCE_ID=your_cos_instance_id
COS_ENDPOINT=your_cos_endpoint
BUCKET_NAME=your_bucket_name

# IBM Watsonx (Standard)
WATSON_STUDIO_PROJECT_ID=your_project_id
WATSONX_API_KEY=your_watsonx_api_key
WATSONX_URL=your_watsonx_url

# IBM Watsonx (LiteLLM)
WATSONX_APIKEY=your_watsonx_apikey  # Required: IBM Cloud API key
WATSONX_URL=your_watsonx_url        # Required: Base URL of your WatsonX instance
WATSONX_PROJECT_ID=your_project_id  # Required: Project ID
WATSONX_TOKEN=optional_iam_token    # Optional: IAM auth token
WATSONX_DEPLOYMENT_SPACE_ID=optional_space_id  # Optional: Deployment space ID
WATSONX_ZENAPIKEY=optional_zen_key  # Optional: Zen API key

# Milvus
MILVUS_GRPC_HOST=your_milvus_host
MILVUS_GRPC_PORT=your_milvus_port
MILVUS_CERT_PATH=path_to_milvus_cert

# Watson Discovery
WATSON_DISCOVERY_API_KEY=your_discovery_api_key
WATSON_DISCOVERY_URL=your_discovery_url
WATSON_DISCOVERY_PROJECT_ID=your_discovery_project_id

# IBM Speech to Text
IBM_STT_API_KEY=your_stt_api_key
IBM_STT_URL=your_stt_url
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
Write-Host "3. You can now use BeeAI framework for building AI agents in this project"