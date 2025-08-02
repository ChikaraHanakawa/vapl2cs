#!/usr/bin/env python

import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob


def process_session_id(session_name):
    """
    Extract base session ID from session name by removing suffix like '_zoom'
    Example: 201_1_2_zoom -> 201_1_2
    """
    # Split by underscore and take the first 3 parts
    parts = session_name.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return session_name


def find_image_directory(session_pattern, image_root_dir):
    """
    Find the image directory for a given session pattern
    Returns the directory path if any images are found
    """
    # Define specific Tabidachi directories to search in
    tabidachi_dirs = [
        "Tabidachi2019-1", 
        "Tabidachi2019-2", 
        "Tabidachi2019-3"
    ]
    
    # Generate patterns for each Tabidachi directory
    patterns = []
    
    for tabidachi_dir in tabidachi_dirs:
        # Pattern: root/Tabidachi2019-X/session_pattern* (direct match in specific directory)
        patterns.append(os.path.join(image_root_dir, tabidachi_dir, f"{session_pattern}*"))
        
        # Pattern: root/Tabidachi2019-X/**/session_pattern* (nested in specific directory)
        patterns.append(os.path.join(image_root_dir, tabidachi_dir, "**", f"{session_pattern}*"))
        
        # Pattern: root/Tabidachi2019-X/**/*session_pattern* (any match in specific directory)
        patterns.append(os.path.join(image_root_dir, tabidachi_dir, "**", f"*{session_pattern}*"))
    
    # Also add direct search for exact session name in each Tabidachi directory
    for tabidachi_dir in tabidachi_dirs:
        patterns.append(os.path.join(image_root_dir, tabidachi_dir, session_pattern))
    
    # First try the specific Tabidachi patterns
    for pattern_idx, pattern in enumerate(patterns):
        matching_dirs = glob.glob(pattern, recursive=True)
        if matching_dirs:
            for dir_path in matching_dirs:
                if os.path.isdir(dir_path):
                    # Check for any image files (*.png)
                    image_files = glob.glob(os.path.join(dir_path, "*.png"))
                    
                    if image_files:
                        # Found images in this directory
                        return dir_path, True
    
    # If we still haven't found anything, search directly in the Tabidachi directories
    tabidachi_dirs = [
        "Tabidachi2019-1", 
        "Tabidachi2019-2", 
        "Tabidachi2019-3"
    ]
    
    for tabidachi_dir in tabidachi_dirs:
        tabidachi_path = os.path.join(image_root_dir, tabidachi_dir)
        
        # Search for directories containing the session pattern
        all_dirs = glob.glob(os.path.join(tabidachi_path, "**"), recursive=True)
        matching_dirs = []
        
        for dir_path in all_dirs:
            if os.path.isdir(dir_path) and session_pattern in os.path.basename(dir_path):
                matching_dirs.append(dir_path)
        
        if matching_dirs:
            for dir_path in matching_dirs:
                # Check for any image files
                image_files = glob.glob(os.path.join(dir_path, "*.png"))
                
                if image_files:
                    return dir_path, True
    
    return None, False


def extract_session_from_audio_path(audio_path):
    """
    Extract session ID from audio path
    Example: /path/to/audio/201_1_2_zoom.wav -> 201_1_2
    """
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    return process_session_id(filename)


def add_image_paths(csv_path, image_root_dir, output_path=None):
    """
    Add image paths to the dataset based on session IDs
    If session column is not available, extract from audio_path
    """
    # If no output path specified, modify the original filename
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_with_images.csv"
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Check if 'session' column exists, if not, create it from audio_path
    has_session_column = 'session' in df.columns
    if not has_session_column:
        if 'audio_path' not in df.columns:
            raise ValueError("CSV must have either 'session' or 'audio_path' column")
        
        print("No 'session' column found, extracting session IDs from audio_path")
        df['session'] = df['audio_path'].apply(extract_session_from_audio_path)
    
    # Create a mapping from session ID to image directory info to avoid redundant searches
    print(f"Processing {len(df)} rows from {csv_path}")
    session_cache = {}
    
    # Add new columns
    df['image_dir'] = None
    df['has_images'] = False
    
    not_found = set()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Finding image paths for {os.path.basename(csv_path)}"):
        session = row['session']
        
        if session not in session_cache:
            # Process the session ID to get the base pattern
            session_pattern = process_session_id(session)
            
            # Find the image directory
            image_dir, has_images = find_image_directory(session_pattern, image_root_dir)
            session_cache[session] = (image_dir, has_images)
            
            if image_dir is None:
                not_found.add(session)
        else:
            image_dir, has_images = session_cache[session]
        
        # Update the dataframe
        df.at[idx, 'image_dir'] = image_dir if image_dir else ""
        df.at[idx, 'has_images'] = has_images
    
    # Save the updated dataframe
    df.to_csv(output_path, index=False)
    
    # Report results
    found_count = sum(1 for s in session_cache.values() if s[0] is not None)
    unique_sessions = len(session_cache)
    
    print(f"Found image directories for {found_count}/{unique_sessions} unique sessions")
    print(f"Missing image directories for {len(not_found)} unique sessions")
    
    if not_found and len(not_found) <= 5:
        print(f"Sessions with missing image directories: {list(not_found)}")
    elif not_found:
        print(f"First 5 sessions with missing image directories: {list(not_found)[:5]}")
    
    print(f"Saved updated dataset to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Add image paths to sliding window datasets")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training dataset CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to the validation dataset CSV")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory to search for image directories")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output files (defaults to same as input)")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional path to the test dataset CSV")
    args = parser.parse_args()
    
    # Create output directory if specified and it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Define output paths
        train_output = os.path.join(args.output_dir, os.path.basename(args.train_csv).replace(".csv", "_with_images.csv"))
        val_output = os.path.join(args.output_dir, os.path.basename(args.val_csv).replace(".csv", "_with_images.csv"))
        test_output = os.path.join(args.output_dir, os.path.basename(args.test_csv).replace(".csv", "_with_images.csv")) if args.test_csv else None
    else:
        train_output = None  # Will use default naming in add_image_paths
        val_output = None
        test_output = None
    
    # Process training dataset
    print("\n===== Processing training dataset =====")
    add_image_paths(args.train_csv, args.image_root, train_output)
    
    # Process validation dataset
    print("\n===== Processing validation dataset =====")
    add_image_paths(args.val_csv, args.image_root, val_output)
    
    # Process test dataset if provided
    if args.test_csv:
        print("\n===== Processing test dataset =====")
        add_image_paths(args.test_csv, args.image_root, test_output)
    
    print("\nAll datasets processed successfully!")


if __name__ == "__main__":
    main()
