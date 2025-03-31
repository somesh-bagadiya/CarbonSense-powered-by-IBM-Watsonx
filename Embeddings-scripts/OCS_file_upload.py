import os
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config

# Load environment variables from .env file
load_dotenv()

# IBM COS Config
COS_API_KEY = os.getenv("COS_API_KEY")
COS_INSTANCE_ID = os.getenv("COS_INSTANCE_ID")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Print to verify (DO NOT print sensitive data in production)
print(f"IBM COS Endpoint: {COS_ENDPOINT}")
print(f"Bucket Name: {BUCKET_NAME}")

# Initialize IBM COS Client
cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY,
    ibm_service_instance_id=COS_INSTANCE_ID,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)

# Define local file paths and corresponding object names in IBM COS
files_to_upload = {
    "/Users/rajatsharma/IBM-CarbonSense-powered-by-Watsonx/Data/dataset_carbon.xlsx"
}

# Upload function
def upload_file(file_path, object_key):
    """Upload a file to IBM COS."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as data:
            cos.put_object(Bucket=BUCKET_NAME, Key=object_key, Body=data)
        print(f"✅ Uploaded: {file_path} -> COS as {object_key}")
    else:
        print(f"❌ File not found: {file_path}")

# Upload all files
for local_path, cos_key in files_to_upload.items():
    upload_file(local_path, cos_key)
    
def list_files():
    """List all files in IBM COS bucket."""
    objects = cos.list_objects_v2(Bucket=BUCKET_NAME)
    if "Contents" in objects:
        for obj in objects["Contents"]:
            print(f"📂 {obj['Key']}")
    else:
        print("⚠️ No files found in the bucket.")

# List uploaded files
list_files()
