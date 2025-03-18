import os
import json
import logging
from io import BytesIO
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from docx import Document  # For extracting text from DOCX files

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# IBM COS Configuration
COS_API_KEY = os.getenv("COS_API_KEY")
COS_INSTANCE_ID = os.getenv("COS_INSTANCE_ID")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
BUCKET_NAME = os.getenv("BUCKET_NAME")
WATSON_STUDIO_PROJECT_ID = os.getenv("WATSON_STUDIO_PROJECT_ID")

# Validate required environment variables
if not all([COS_API_KEY, COS_INSTANCE_ID, COS_ENDPOINT, BUCKET_NAME]):
    logging.error("Missing required IBM COS environment variables.")
    exit(1)

logging.info(f"IBM COS Endpoint: {COS_ENDPOINT}")
logging.info(f"Bucket Name: {BUCKET_NAME}")

# Initialize IBM COS Client
try:
    cos_client = ibm_boto3.client(
        "s3",
        ibm_api_key_id=COS_API_KEY,
        ibm_service_instance_id=COS_INSTANCE_ID,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT
    )
    logging.info("IBM COS client initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing IBM COS client: {e}")
    exit(1)

# Initialize IBM Watsonx.ai Client and Embedding Model
try:
    client = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=COS_API_KEY
    )
    embedding = Embeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
        project_id=WATSON_STUDIO_PROJECT_ID,
        credentials=client
    )
    logging.info("IBM Watsonx embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Error initializing Watsonx.ai model: {e}")
    exit(1)

def extract_text_from_docx(file_bytes):
    """Extract text from a DOCX file given its bytes."""
    try:
        doc = Document(BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error extracting DOCX text: {e}")
        raise

def chunk_text(text, max_chars=1000):
    """
    Naively chunk text into pieces that do not exceed max_chars.
    Adjust this logic as needed to better match token limits.
    """
    lines = text.splitlines()
    chunks = []
    current_chunk = ""
    for line in lines:
        # Add line if it does not exceed the maximum length
        if len(current_chunk) + len(line) + 1 <= max_chars:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# List all objects in the bucket
try:
    response = cos_client.list_objects_v2(Bucket=BUCKET_NAME)
    contents = response.get("Contents")
    if not contents:
        logging.warning("No files found in the specified IBM COS bucket.")
        exit(0)
except Exception as e:
    logging.error(f"Error listing objects in bucket: {e}")
    exit(1)

# Process each file in the bucket
for obj in contents:
    object_name = obj["Key"]
    logging.info(f"Processing file: {object_name}")

    try:
        # Download file from COS
        file_obj = cos_client.get_object(Bucket=BUCKET_NAME, Key=object_name)
    except Exception as e:
        logging.error(f"Error retrieving file {object_name}: {e}")
        continue

    # Determine how to extract text based on file extension
    file_text = ""
    lower_name = object_name.lower()
    try:
        if lower_name.endswith('.docx'):
            # For DOCX, extract text using python-docx
            file_bytes = file_obj["Body"].read()
            file_text = extract_text_from_docx(file_bytes)
        else:
            # Assume other files are text-based (CSV, TXT, etc.)
            file_text = file_obj["Body"].read().decode("utf-8")
    except Exception as e:
        logging.error(f"Error processing file {object_name}: {e}")
        continue

    # Check if text needs chunking (e.g., if it's too long)
    # Here, we use a simple character count threshold. You might refine this based on tokenization.
    if len(file_text) > 3000:
        chunks = chunk_text(file_text, max_chars=1000)
        logging.info(f"File {object_name} is large; splitting into {len(chunks)} chunks.")
    else:
        chunks = [file_text]

    # Generate embeddings for the text (or chunks)
    try:
        # The embed_documents method takes a list of texts
        embedding_result = embedding.embed_documents(chunks)
        # If multiple chunks, store embeddings for each chunk
        embeddings = embedding_result
        logging.info(f"Embeddings generated for {object_name}.")
    except Exception as e:
        logging.error(f"Error generating embeddings for {object_name}: {e}")
        continue

    # Prepare and save embeddings to a JSON file
    embedding_data = {
        "file_name": object_name,
        "chunks": chunks,
        "embeddings": embeddings
    }
    local_embedding_file = f"embeddings_{object_name.replace('/', '_')}.json"
    try:
        with open(local_embedding_file, "w") as f:
            json.dump(embedding_data, f)
        logging.info(f"Embeddings saved locally: {local_embedding_file}")
    except Exception as e:
        logging.error(f"Error saving embeddings locally for {object_name}: {e}")
        continue

    # Upload embeddings JSON file to IBM COS
    embedding_cos_key = f"embeddings/{object_name.replace('/', '_')}.json"
    try:
        cos_client.upload_file(Filename=local_embedding_file, Bucket=BUCKET_NAME, Key=embedding_cos_key)
        logging.info(f"Embeddings uploaded to COS: {embedding_cos_key}")
    except Exception as e:
        logging.error(f"Error uploading embeddings to COS for {object_name}: {e}")
        continue

logging.info("Embedding processing completed successfully.")
