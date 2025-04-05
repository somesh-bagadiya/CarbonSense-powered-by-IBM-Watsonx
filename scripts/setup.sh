#!/bin/bash

echo "Setting up CarbonSense environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    cat > .env << EOL
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
EOL
    echo "Please update the .env file with your credentials"
fi

# Create data directories if they don't exist
echo "Creating data directories..."
mkdir -p Data_RAW
mkdir -p Data_processed/Global
mkdir -p "Data_processed/United States of America"

echo -e "\nSetup complete! Next steps:"
echo "1. Update the .env file with your credentials"
echo "2. Run 'python -m carbonsense.main --mode generate' to generate embeddings"
echo "3. Run 'uvicorn carbonsense.api.app:app --reload' to start the API server" 