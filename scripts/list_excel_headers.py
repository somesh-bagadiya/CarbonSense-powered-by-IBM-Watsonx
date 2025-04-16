#!/usr/bin/env python
"""
Script to list headers (column names) of all Excel files in the project directory.
"""

import os
import pandas as pd
from pathlib import Path

def get_excel_files(directory):
    """Find all Excel files in the given directory and its subdirectories."""
    excel_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.xlsx', '.xls')):
                excel_files.append(os.path.join(root, file))
    return excel_files

def get_excel_headers(file_path):
    """Extract headers from an Excel file."""
    try:
        # Get all sheet names
        excel = pd.ExcelFile(file_path)
        sheet_names = excel.sheet_names
        
        headers_by_sheet = {}
        
        for sheet in sheet_names:
            try:
                # Read only the header row
                df = pd.read_excel(file_path, sheet_name=sheet, nrows=0)
                headers_by_sheet[sheet] = list(df.columns)
            except Exception as e:
                headers_by_sheet[sheet] = f"Error reading sheet: {str(e)}"
                
        return headers_by_sheet
    
    except Exception as e:
        return {"Error": f"Could not read file: {str(e)}"}

def main():
    # Get the project root directory (assuming this script is in the scripts folder)
    project_dir = Path(__file__).parent.parent
    
    # Find all Excel files
    excel_files = get_excel_files(project_dir)
    
    # Create output file path
    output_file = project_dir / "excel_headers.txt"
    
    # Open file for writing
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Found {len(excel_files)} Excel files.\n")
        f.write("=" * 80 + "\n")
        
        # Process each file
        for file_path in excel_files:
            rel_path = os.path.relpath(file_path, project_dir)
            f.write(f"\nFile: {rel_path}\n")
            f.write("-" * 80 + "\n")
            
            headers = get_excel_headers(file_path)
            
            for sheet, columns in headers.items():
                f.write(f"\nSheet: {sheet}\n")
                if isinstance(columns, list):
                    for i, col in enumerate(columns, 1):
                        f.write(f"{i}. {col}\n")
                else:
                    f.write(f"{columns}\n")  # Write error message
            
            f.write("=" * 80 + "\n")
    
    print(f"Excel headers have been exported to: {output_file}")

if __name__ == "__main__":
    main()
