import os
import shutil
import pandas as pd
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndustryDataProcessor:
    def __init__(self):
        self.regions = {
            'Global': ['wood.xlsx', 'water.xlsx', 'transport.xlsx', 'textile.xlsx', 
                      'scope 1 & downsrtream.xlsx', 'processing metals.xlsx', 'plastics.xlsx',
                      'paper & packaging.xlsx', 'metals, non-ferro.xlsx', 'metals, ferro.xlsx',
                      'laminates.xlsx', 'heat.xlsx', 'glass.xlsx', 'fuels.xlsx', 'food.xlsx',
                      'fibres.xlsx', 'end-of-life.xlsx', 'electronics.xlsx', 'electricity general industry.xlsx',
                      'chemicals.xlsx', 'chem proxi.xlsx', 'ceramics.xlsx', 'building materials.xlsx',
                      'agriculture.xlsx', '.heading.xlsx', 'electricity Rest of the World.xlsx',
                      'electricity EU .xlsx', 'electricity China.xlsx', 'electricity India.xlsx',
                      'electricity Canada.xlsx'],
            'United States of America': ['electricity USA.xlsx']
        }
        
        self.base_raw_path = 'Data_RAW/industries'
        self.base_processed_path = 'Data_processed'
        
    def create_directory_structure(self):
        """Create the directory structure for processed data."""
        try:
            # Only create region directories if they don't exist
            for region in self.regions.keys():
                region_path = os.path.join(self.base_processed_path, region)
                if not os.path.exists(region_path):
                    os.makedirs(region_path)
                    logger.info(f"Created directory: {region}")
                    
        except Exception as e:
            logger.error(f"Error creating directory structure: {str(e)}")
            raise
            
    def process_and_organize_files(self):
        """Process and organize files by region."""
        try:
            for region, files in self.regions.items():
                region_path = os.path.join(self.base_processed_path, region)
                
                for file in files:
                    source_path = os.path.join(self.base_raw_path, file)
                    if not os.path.exists(source_path):
                        logger.warning(f"File not found: {source_path}")
                        continue
                        
                    # Copy and process the file
                    self._process_file(source_path, region_path, file)
                    
            logger.info("Files processed and organized successfully")
            
        except Exception as e:
            logger.error(f"Error processing and organizing files: {str(e)}")
            raise
            
    def _process_file(self, source_path: str, target_dir: str, filename: str):
        """Process a single file and save it to the target directory."""
        try:
            # Read the Excel file
            df = pd.read_excel(source_path)
            
            # Clean column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Add metadata columns
            df['source_file'] = filename
            df['region'] = os.path.basename(target_dir)
            
            # Save processed file
            target_path = os.path.join(target_dir, filename)
            df.to_excel(target_path, index=False)
            
            logger.info(f"Processed file: {filename}")
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            raise
            
    def create_consolidated_files(self):
        """Create consolidated files for each region."""
        try:
            for region in self.regions.keys():
                region_path = os.path.join(self.base_processed_path, region)
                consolidated_data = []
                
                # Read all files in the region
                for file in os.listdir(region_path):
                    if file.endswith('.xlsx') and not file.startswith('consolidated_'):
                        file_path = os.path.join(region_path, file)
                        df = pd.read_excel(file_path)
                        consolidated_data.append(df)
                
                # Combine all dataframes
                if consolidated_data:
                    combined_df = pd.concat(consolidated_data, ignore_index=True)
                    
                    # Save consolidated file
                    consolidated_path = os.path.join(region_path, f'consolidated_{region.lower()}.xlsx')
                    combined_df.to_excel(consolidated_path, index=False)
                    
                    logger.info(f"Created consolidated file for region: {region}")
                    
        except Exception as e:
            logger.error(f"Error creating consolidated files: {str(e)}")
            raise

def main():
    processor = IndustryDataProcessor()
    
    # Create directory structure
    processor.create_directory_structure()
    
    # Process and organize files
    processor.process_and_organize_files()
    
    # Create consolidated files
    processor.create_consolidated_files()
    
    logger.info("Industry data processing completed successfully")

if __name__ == "__main__":
    main() 