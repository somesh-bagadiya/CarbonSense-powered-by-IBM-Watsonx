#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script processes Excel files from backup-dataset and creates a cleaned version in data_processed.
It keeps only the columns specified in shortlist_of_columns.txt.
"""

import os
import pandas as pd
import shutil
import logging
import openpyxl
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("dataset_processing.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Define the base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, "backup-dataset")
TARGET_DIR = os.path.join(BASE_DIR, "Data_processed")

# Define the columns to keep (from shortlist_of_columns.txt) with units included
COLUMNS_TO_KEEP = [
    "industry", "unit", "process", 
    "carbon (kg CO2 eq)", "ced (MJ)", 
    "global warming (kg CO2 eq)", "climate change (kg CO2 eq)", 
    "region"
]

# Mapping for original column names to new ones with units
COLUMN_RENAME_MAP = {
    "carbon": "carbon (kg CO2 eq)",
    "ced": "ced (MJ)",
    "global warming,": "global warming (kg CO2 eq)",
    "climate change": "climate change (kg CO2 eq)"
}

# Column with variations (capitalization, spelling, spacing, etc.)
COLUMN_VARIATIONS = {
    "industry": ["industry", "Industry", "industries", "Industries", "industrie"],
    "unit": ["unit", "Unit", "units", "Units", "measure", "Measure"],
    "process": ["process", "Process", "processes", "Processes", "processing", "Processing"],
    "carbon (kg CO2 eq)": ["carbon", "Carbon", "carbon footprint", "Carbon footprint", "co2", "CO2", "CO2 eq", "co2 eq"],
    "ced (MJ)": ["ced", "CED", "cumulative energy demand", "Cumulative energy demand", "cumulative_energy_demand", "energy demand"],
    "global warming (kg CO2 eq)": ["global warming,", "Global warming,", "global warming", "Global warming", "global warming potential", "GWP", "gwp", "warming"],
    "climate change (kg CO2 eq)": ["climate change", "Climate change", "climate_change", "Climate_change", "climate impact", "Climate impact"],
    "region": ["region", "Region", "regions", "Regions", "country", "Country", "location", "Location"]
}

def verify_excel_files():
    """Test function to verify Excel files can be read properly and count rows"""
    logger.info("Starting Excel file verification...")
    excel_files = []
    
    # Check electricity files
    electricity_source = os.path.join(SOURCE_DIR, "electricity")
    if os.path.exists(electricity_source):
        for file in os.listdir(electricity_source):
            if file.endswith('.xlsx') and not file.startswith('~$'):
                excel_files.append(os.path.join(electricity_source, file))
                
    # Check industry files
    industry_source = os.path.join(SOURCE_DIR, "industry")
    if os.path.exists(industry_source):
        for file in os.listdir(industry_source):
            if file.endswith('.xlsx') and not file.startswith('~$'):
                excel_files.append(os.path.join(industry_source, file))
    
    # Try to read each file and print columns and row count
    success_count = 0
    fail_count = 0
    row_count_data = []
    
    for file_path in excel_files:
        try:
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            num_rows = len(df)
            num_cols = len(df.columns)
            logger.info(f"Successfully read: {os.path.basename(file_path)}")
            logger.info(f"Columns ({num_cols}): {list(df.columns)}")
            logger.info(f"Number of rows: {num_rows}")
            row_count_data.append((os.path.basename(file_path), num_rows, num_cols))
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to read: {os.path.basename(file_path)}")
            logger.error(f"Error: {str(e)}")
            fail_count += 1
    
    # Print summary
    logger.info("===== Summary =====")
    logger.info(f"Verification complete. Successfully read {success_count} files. Failed to read {fail_count} files.")
    
    if row_count_data:
        logger.info("\nRow count by file:")
        for file_name, rows, cols in sorted(row_count_data, key=lambda x: x[1], reverse=True):
            logger.info(f"{file_name}: {rows} rows, {cols} columns")
        
        total_rows = sum(item[1] for item in row_count_data)
        logger.info(f"\nTotal rows across all files: {total_rows}")
    
    return success_count, fail_count, row_count_data

def ensure_directories(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def list_excel_files(directory):
    """List all Excel files in a directory and its subdirectories"""
    excel_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.xlsx') and not file.startswith('~$'):
                excel_files.append(os.path.join(root, file))
    return excel_files

def get_matching_columns(df):
    """Find matching columns between DataFrame and our list of columns to keep using exact and fuzzy matching"""
    matched_columns = {}
    
    # First try exact matches
    for col_key in COLUMNS_TO_KEEP:
        variations = COLUMN_VARIATIONS.get(col_key.lower(), [col_key])
        found = False
        
        # Try exact matches first
        for variation in variations:
            if variation in df.columns:
                matched_columns[col_key] = variation
                found = True
                break
                
        # If exact match not found, try case-insensitive match
        if not found:
            for df_col in df.columns:
                for variation in variations:
                    if df_col.lower() == variation.lower():
                        matched_columns[col_key] = df_col
                        found = True
                        break
                if found:
                    break
                    
        # If still not found, try substring matching
        if not found:
            for df_col in df.columns:
                for variation in variations:
                    # Check if variation is a significant part of the column name or vice versa
                    if (variation.lower() in df_col.lower() or 
                        df_col.lower() in variation.lower()):
                        matched_columns[col_key] = df_col
                        found = True
                        logger.info(f"Fuzzy matched '{col_key}' to column '{df_col}'")
                        break
                if found:
                    break
        
        if not found:
            logger.warning(f"Column {col_key} not found in DataFrame")
    
    return matched_columns

def process_excel_file(source_file, target_file):
    """Process a single Excel file"""
    try:
        # Read Excel file
        df = pd.read_excel(source_file, sheet_name="Sheet1")
        
        # Print column headers for debugging
        logger.info(f"File: {os.path.basename(source_file)}")
        logger.info(f"Original columns: {list(df.columns)}")
        logger.info(f"Original row count: {len(df)}")
        
        # Get matching columns
        matched_columns = get_matching_columns(df)
        
        if not matched_columns:
            logger.error(f"No matching columns found in {source_file}")
            return
        
        logger.info(f"Matched columns: {matched_columns}")
        
        # Select only the matched columns
        df_filtered = df[[matched_columns[col] for col in matched_columns if matched_columns.get(col) in df.columns]]
        
        # Create rename dictionary
        rename_dict = {}
        for col in matched_columns:
            source_col = matched_columns[col]
            # Check if this is a column that needs unit in its name
            if col in COLUMN_RENAME_MAP.values():
                # Already has unit in the name
                rename_dict[source_col] = col
            elif col in COLUMN_RENAME_MAP:
                # Need to add unit to the name
                rename_dict[source_col] = COLUMN_RENAME_MAP[col]
            else:
                # Regular column
                rename_dict[source_col] = col
                
        # Rename columns to standardized names with units
        df_filtered = df_filtered.rename(columns=rename_dict)
        
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        
        # Add data types as header row 
        # Create a dictionary for data types based on shortlist_of_columns.txt
        # Note: We don't need to include units in data types since they're now in column names
        datatype_dict = {
            "industry": "Categorical (text)",
            "unit": "Various (e.g. kg, kWh)",
            "process": "Categorical (text)",
            "carbon (kg CO2 eq)": "Carbon footprint",
            "ced (MJ)": "Cumulative energy demand",
            "global warming (kg CO2 eq)": "Global warming potential",
            "climate change (kg CO2 eq)": "Climate change impact",
            "region": "Categorical (text)"
        }
        
        # Save with column headers including datatype
        with pd.ExcelWriter(target_file, engine='openpyxl') as writer:
            df_filtered.to_excel(writer, index=False, sheet_name="Sheet1")
            
            # Add data type information to the header
            worksheet = writer.sheets["Sheet1"]
            for i, col in enumerate(df_filtered.columns, start=1):
                datatype = datatype_dict.get(col, "")
                if datatype:
                    cell = worksheet.cell(row=1, column=i)
                    cell.comment = openpyxl.comments.Comment(f"Data type: {datatype}", "Data Processor")
        
        logger.info(f"Processed: {os.path.basename(source_file)} -> {os.path.basename(target_file)}")
        logger.info(f"Processed row count: {len(df_filtered)}")
    except Exception as e:
        logger.error(f"Error processing {source_file}: {str(e)}")

def main():
    """Main function to process all Excel files"""
    logger.info("Starting dataset preprocessing")
    
    # Ensure target directories exist
    ensure_directories(TARGET_DIR)
    ensure_directories(os.path.join(TARGET_DIR, "electricity"))
    ensure_directories(os.path.join(TARGET_DIR, "industry"))
    
    # Process electricity files
    electricity_source = os.path.join(SOURCE_DIR, "electricity")
    electricity_target = os.path.join(TARGET_DIR, "electricity")
    
    if os.path.exists(electricity_source):
        electricity_files = list_excel_files(electricity_source)
        logger.info(f"Found {len(electricity_files)} electricity files to process")
        
        for source_file in electricity_files:
            file_name = os.path.basename(source_file)
            target_file = os.path.join(electricity_target, file_name)
            process_excel_file(source_file, target_file)
    else:
        logger.warning(f"Directory not found: {electricity_source}")
    
    # Process industry files
    industry_source = os.path.join(SOURCE_DIR, "industry")
    industry_target = os.path.join(TARGET_DIR, "industry")
    
    if os.path.exists(industry_source):
        industry_files = list_excel_files(industry_source)
        logger.info(f"Found {len(industry_files)} industry files to process")
        
        for source_file in industry_files:
            file_name = os.path.basename(source_file)
            target_file = os.path.join(industry_target, file_name)
            process_excel_file(source_file, target_file)
    else:
        logger.warning(f"Directory not found: {industry_source}")
    
    # Process any other subdirectories if needed
    for subdir in ["metadata", "regional"]:
        source_subdir = os.path.join(SOURCE_DIR, subdir)
        target_subdir = os.path.join(TARGET_DIR, subdir)
        
        if os.path.exists(source_subdir):
            ensure_directories(target_subdir)
            files = list_excel_files(source_subdir)
            logger.info(f"Found {len(files)} files in {subdir} to process")
            
            for source_file in files:
                rel_path = os.path.relpath(source_file, source_subdir)
                target_file = os.path.join(target_subdir, rel_path)
                ensure_directories(os.path.dirname(target_file))
                process_excel_file(source_file, target_file)
        else:
            logger.warning(f"Directory not found: {source_subdir}")
    
    logger.info("Dataset preprocessing completed")

if __name__ == "__main__":
    # If --verify flag is passed, only run verification
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        logger.info("Running in verification mode only")
        verify_excel_files()
    else:
        # Run the full processing
        main()
