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
from docx import Document
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# Load .env values
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# IBM COS Config
COS_API_KEY = os.getenv("COS_API_KEY")
COS_INSTANCE_ID = os.getenv("COS_INSTANCE_ID")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
BUCKET_NAME = os.getenv("BUCKET_NAME")
WATSON_STUDIO_PROJECT_ID = os.getenv("WATSON_STUDIO_PROJECT_ID")

# Milvus Config
# MILVUS_HOST = os.getenv("MILVUS_REST_HOST")
# MILVUS_PORT = os.getenv("MILVUS_REST_PORT")
MILVUS_HOST = os.getenv("MILVUS_GRPC_HOST")
MILVUS_PORT = os.getenv("MILVUS_GRPC_PORT")
print(MILVUS_HOST, MILVUS_PORT, type(MILVUS_HOST), type(MILVUS_PORT))

# IBM User Config
IBM_USERNAME = os.getenv("IBM_USERNAME")
IBM_PASSWORD = os.getenv("IBM_PASSWORD")

if not all([COS_API_KEY, COS_INSTANCE_ID, COS_ENDPOINT, BUCKET_NAME]):
    logging.error("Missing required environment variables.")
    exit(1)

# Connect to IBM COS
cos_client = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_INSTANCE_ID,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

# Connect to Milvus
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, user="ibmlhapikey", password=COS_API_KEY, secure=True)

# Init Watsonx Embedding Model
client = Credentials(url="https://us-south.ml.cloud.ibm.com", api_key=COS_API_KEY)
embedding = Embeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
    project_id=WATSON_STUDIO_PROJECT_ID,
    credentials=client
)

def extract_text_from_docx(file_bytes):
    doc = Document(BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])

def chunk_text(text, max_chars=400):  # Reduced to stay within token limits
    lines = text.splitlines()
    chunks = []
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 <= max_chars:
            current_chunk += line + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def create_milvus_collection(collection_name: str, dim: int):
    if collection_name in Collection.list():
        return Collection(name=collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500)
    ]
    schema = CollectionSchema(fields, description="CarbonSense document embeddings")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}})
    collection.load()
    return collection

# Process files from COS
response = cos_client.list_objects_v2(Bucket=BUCKET_NAME)
contents = response.get("Contents", [])
if not contents:
    logging.warning("No files found in bucket.")
    exit(0)

for obj in contents:
    object_name = obj["Key"]

    # 🚫 Skip already-processed JSON
    if object_name.endswith(".json"):
        logging.info(f"Skipping JSON: {object_name}")
        continue

    logging.info(f"Processing file: {object_name}")

    try:
        file_obj = cos_client.get_object(Bucket=BUCKET_NAME, Key=object_name)
        lower_name = object_name.lower()

        if lower_name.endswith(".docx"):
            file_bytes = file_obj["Body"].read()
            file_text = extract_text_from_docx(file_bytes)
        else:
            file_text = file_obj["Body"].read().decode("utf-8")

        chunks = chunk_text(file_text, max_chars=400)
        if len(chunks) > 1:
            logging.info(f"File {object_name} split into {len(chunks)} chunks.")

        embeddings = embedding.embed_documents(chunks)
        logging.info(f"Embeddings generated for {object_name}.")

        # Save locally as backup
        embedding_data = {
            "file_name": object_name,
            "chunks": chunks,
            "embeddings": embeddings
        }
        local_file = f"embeddings_{object_name.replace('/', '_')}.json"
        with open(local_file, "w") as f:
            json.dump(embedding_data, f)
        logging.info(f"Saved locally: {local_file}")

        # Upload to COS
        cos_key = f"embeddings/{object_name.replace('/', '_')}.json"
        cos_client.upload_file(Filename=local_file, Bucket=BUCKET_NAME, Key=cos_key)
        logging.info(f"Uploaded to COS: {cos_key}")

        # ➕ Insert into Milvus
        dim = len(embeddings[0])
        collection = create_milvus_collection("carbon_embeddings", dim)

        insert_data = [
            embeddings,                  # embeddings (List[List[float]])
            chunks,                      # chunk_texts
            [object_name] * len(chunks)  # file_name
        ]
        collection.insert(insert_data)
        logging.info(f"Inserted {len(chunks)} vectors into Milvus for {object_name}.")

    except Exception as e:
        logging.error(f"Error processing {object_name}: {e}")
        continue

logging.info("All files processed successfully.")
