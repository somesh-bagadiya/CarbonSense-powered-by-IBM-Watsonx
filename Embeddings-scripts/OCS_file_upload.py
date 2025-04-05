import os
import logging
from typing import Dict, List, Optional
from pathlib import Path
from dotenv import load_dotenv
import ibm_boto3
from ibm_botocore.client import Config

class ConfigManager:
    """Manages configuration and environment variables."""
    
    def __init__(self):
        load_dotenv()
        self._validate_environment()
        
    def _validate_environment(self) -> None:
        """Validates required environment variables."""
        required_vars = {
            "COS_API_KEY": "IBM Cloud Object Storage API Key",
            "COS_INSTANCE_ID": "IBM Cloud Object Storage Instance ID",
            "COS_ENDPOINT": "IBM Cloud Object Storage Endpoint",
            "BUCKET_NAME": "IBM Cloud Object Storage Bucket Name"
        }
        
        missing_vars = [var for var, desc in required_vars.items() if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

class COSManager:
    """Manages IBM Cloud Object Storage operations."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.client = self._init_cos_client()
        self.bucket_name = os.getenv("BUCKET_NAME")
        
    def _init_cos_client(self) -> Any:
        """Initializes IBM Cloud Object Storage client."""
        return ibm_boto3.client(
            "s3",
            ibm_api_key_id=os.getenv("COS_API_KEY"),
            ibm_service_instance_id=os.getenv("COS_INSTANCE_ID"),
            config=Config(signature_version="oauth"),
            endpoint_url=os.getenv("COS_ENDPOINT")
        )
    
    def upload_file(self, file_path: str, object_key: str) -> bool:
        """Uploads a file to IBM COS.
        
        Args:
            file_path: Local path to the file
            object_key: Key to store the file as in COS
            
        Returns:
            True if upload was successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                return False
                
            with open(file_path, "rb") as data:
                self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=data
                )
            logging.info(f"Uploaded: {file_path} -> COS as {object_key}")
            return True
            
        except Exception as e:
            logging.error(f"Error uploading {file_path}: {str(e)}")
            return False
    
    def list_files(self) -> List[Dict[str, str]]:
        """Lists all files in the COS bucket.
        
        Returns:
            List of dictionaries containing file information
        """
        try:
            objects = self.client.list_objects_v2(Bucket=self.bucket_name)
            if "Contents" not in objects:
                logging.warning("No files found in the bucket")
                return []
                
            return [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat()
                }
                for obj in objects["Contents"]
            ]
            
        except Exception as e:
            logging.error(f"Error listing files: {str(e)}")
            return []

class FileUploader:
    """Manages file upload operations."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.cos = COSManager(config)
        
    def upload_files(self, files_to_upload: Dict[str, str]) -> None:
        """Uploads multiple files to COS.
        
        Args:
            files_to_upload: Dictionary mapping local paths to COS object keys
        """
        success_count = 0
        total_files = len(files_to_upload)
        
        for local_path, cos_key in files_to_upload.items():
            if self.cos.upload_file(local_path, cos_key):
                success_count += 1
                
        logging.info(f"Uploaded {success_count} out of {total_files} files successfully")
    
    def list_uploaded_files(self) -> None:
        """Lists all files in the COS bucket."""
        files = self.cos.list_files()
        if files:
            logging.info("Files in bucket:")
            for file_info in files:
                logging.info(f"📂 {file_info['key']} (Size: {file_info['size']} bytes)")
        else:
            logging.warning("No files found in the bucket")

def main():
    """Main execution function."""
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize configuration
        config = ConfigManager()
        
        # Define files to upload
        files_to_upload = {
            str(Path("Data/NAICS_byGHG_CO2e_USD2022/Aboutv1.3SupplyChainGHGEmissionFactors.docx")): 
                "documents/Aboutv1.3SupplyChainGHGEmissionFactors.docx",
            str(Path("Data/NAICS_byGHG_CO2e_USD2022/SupplyChainGHGEmissionFactors_v1.3.0_NAICS_byGHG_USD2022.csv")): 
                "documents/SupplyChainGHGEmissionFactors_v1.3.0_NAICS_byGHG_USD2022.csv",
            str(Path("Data/NAICS_byGHG_CO2e_USD2022/SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv")): 
                "documents/SupplyChainGHGEmissionFactors_v1.3.0_NAICS_CO2e_USD2022.csv"
        }
        
        # Initialize uploader
        uploader = FileUploader(config)
        
        # Upload files
        uploader.upload_files(files_to_upload)
        
        # List uploaded files
        uploader.list_uploaded_files()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
