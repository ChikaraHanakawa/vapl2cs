#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import glob
import argparse

def filter_csv_files(input_dir, output_dir, file_pattern="*.csv"):
    """
    Read CSV files, filter out rows where has_images is False,
    and save the filtered data to new CSV files.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing input CSV files
    output_dir : str
        Directory to save output CSV files
    file_pattern : str, optional
        Pattern to match CSV files, default is "*.csv"
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir} matching pattern {file_pattern}")
        return
    
    for csv_file in csv_files:
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file)
            
            # Check if the required column exists
            if 'has_images' not in df.columns:
                print(f"Warning: 'has_images' column not found in {csv_file}. Skipping file.")
                continue
            
            # Filter rows where has_images is not False
            # Convert to boolean if it's string
            if df['has_images'].dtype == 'object':
                # Handle various string representations of True/False
                df['has_images'] = df['has_images'].map(lambda x: str(x).lower() in ['true', '1', 't', 'yes'])
            
            filtered_df = df[df['has_images'] == True]
            
            # Get the basename of the input file
            base_name = os.path.basename(csv_file)
            output_path = os.path.join(output_dir, f"filtered_{base_name}")
            
            # Save the filtered DataFrame
            filtered_df.to_csv(output_path, index=False)
            
            print(f"Processed {csv_file}: {len(df) - len(filtered_df)} rows removed, {len(filtered_df)} rows kept.")
            print(f"Saved filtered data to {output_path}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Filter CSV files based on 'has_images' column")
    parser.add_argument('--input', type=str, required=True, help='Input directory containing CSV files')
    parser.add_argument('--output', type=str, required=True, help='Output directory for filtered CSV files')
    parser.add_argument('--pattern', type=str, default='*.csv', help='File pattern to match CSV files')
    
    args = parser.parse_args()
    
    filter_csv_files(args.input, args.output, args.pattern)

if __name__ == "__main__":
    main()
