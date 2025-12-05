"""
XLS to CSV Converter Script

Converts all .xls files from xls subdirectories to CSV format
and saves them in corresponding csv subdirectories.
"""

import pandas as pd
import os
from pathlib import Path

# Base directory
BASE_DIR = Path('data')

# Folders to process
FOLDERS = [
    'Directional_Tyre',
    'No_Pattern',
    'No_Tyre_Control',
    'Symmetrical_Tyre'
]

def convert_xls_to_csv(xls_path, csv_path):
    """
    Convert a LabVIEW .xls file (tab-delimited text) to CSV format.
    
    Args:
        xls_path: Path to input .xls file (LabVIEW format)
        csv_path: Path to output .csv file
    """
    try:
        # Read the file to find where header ends
        with open(xls_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find the line with "***End_of_Header***"
        header_end = 0
        for i, line in enumerate(lines):
            if '***End_of_Header***' in line:
                header_end = i + 1
                break
        
        # Skip empty lines after header
        while header_end < len(lines) and lines[header_end].strip() == '':
            header_end += 1
        
        # Read the data starting after the header
        # LabVIEW files are tab-delimited
        df = pd.read_csv(xls_path, sep='\t', skiprows=header_end, encoding='utf-8', encoding_errors='ignore')
        
        # Remove any completely empty rows or columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        
        print(f"[OK] Converted: {xls_path.name} -> {csv_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Error converting {xls_path.name}: {str(e)}")
        return False

def process_folder(folder_name):
    """
    Process all XLS files in a folder's xls subdirectory.
    
    Args:
        folder_name: Name of the folder to process
    """
    xls_dir = BASE_DIR / folder_name / 'xls'
    csv_dir = BASE_DIR / folder_name / 'csv'
    
    # Check if directories exist
    if not xls_dir.exists():
        print(f"[WARNING] {xls_dir} does not exist. Skipping...")
        return
    
    # Create CSV directory if it doesn't exist
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all XLS files
    xls_files = list(xls_dir.glob('*.xls'))
    
    if not xls_files:
        print(f"[INFO] No .xls files found in {xls_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing: {folder_name}")
    print(f"{'='*60}")
    print(f"Source: {xls_dir}")
    print(f"Destination: {csv_dir}")
    print(f"Files to convert: {len(xls_files)}")
    print(f"{'-'*60}")
    
    # Convert each file
    success_count = 0
    for xls_file in sorted(xls_files):
        csv_file = csv_dir / (xls_file.stem + '.csv')
        if convert_xls_to_csv(xls_file, csv_file):
            success_count += 1
    
    print(f"{'-'*60}")
    print(f"Successfully converted: {success_count}/{len(xls_files)} files")

def main():
    """Main function to process all folders."""
    print("="*60)
    print("XLS to CSV Converter")
    print("="*60)
    
    # Check if base directory exists
    if not BASE_DIR.exists():
        print(f"[ERROR] Base directory '{BASE_DIR}' does not exist!")
        return
    
    # Process each folder
    total_folders = len(FOLDERS)
    for idx, folder in enumerate(FOLDERS, 1):
        process_folder(folder)
    
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"Processed {total_folders} folders")
    print("="*60)

if __name__ == "__main__":
    main()

