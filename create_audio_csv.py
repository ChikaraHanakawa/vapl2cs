#!/usr/bin/env python

import os
import sys
import argparse
import pandas as pd
import random
from pathlib import Path
from os.path import join, isfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add parent directory to path to make imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules directly, since some don't have a main() function
import vap.data.create_sliding_window_dset as sliding_window_module
from vap.data.datamodule import VAPDataModule
from vap.data.dset_event import create_classification_dset


# Recreate the functionality from create_audio_vad_csv.py as a function
def create_audio_vad_csv(args):
    """
    Create a CSV file mapping audio files to VAD JSON files.
    Based on functionality in create_audio_vad_csv.py
    """
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    audio_paths = list(Path(args.audio_dir).rglob("*.wav"))
    data = []
    skipped = []
    for audio_path in tqdm(audio_paths):
        name = audio_path.stem
        vad_path = join(args.vad_dir, f"{name}.json")
        if not isfile(vad_path):
            skipped.append(vad_path)
            continue
        data.append(
            {
                "audio_path": str(audio_path),
                "vad_path": vad_path,
            }
        )

    if len(skipped) > 0:
        print("Skipped: ", len(skipped))
        with open("/tmp/create_audio_vad_json_errors.txt", "w") as f:
            f.write("\n".join(skipped))
        print("See -> /tmp/create_audio_vad_json_errors.txt")
        print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print("Saved -> ", args.output)


def parse_args():
    parser = argparse.ArgumentParser(description="Process VAP data workflow")
    
    # Basic directory settings
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--vad_dir", type=str, required=True, help="Directory containing VAD JSON files")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory for output files")
    
    # Sliding window parameters
    parser.add_argument("--duration", type=float, default=20, help="Duration of sliding window in seconds")
    parser.add_argument("--overlap", type=float, default=5, help="Overlap of sliding windows in seconds")
    parser.add_argument("--horizon", type=float, default=2, help="Prediction horizon in seconds")
    
    # Dataset split parameters
    parser.add_argument("--val_size", type=float, default=0.05, help="Validation set size ratio")
    parser.add_argument("--test_size", type=float, default=0.15, help="Test set size ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Event extraction parameters
    parser.add_argument("--pre_cond_time", type=float, default=1.0, help="Single speaker time prior to silence")
    parser.add_argument("--post_cond_time", type=float, default=2.0, help="Single speaker time post silence")
    parser.add_argument("--min_silence_time", type=float, default=0.1, help="Minimum reaction time / silence duration")
    parser.add_argument("--ipu_based_events", action="store_true", help="Use IPU-based events instead of HoldShift")
    
    # DataModule parameters for checking data
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for data check")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    
    # Control flags
    parser.add_argument("--skip_audio_vad_csv", action="store_true", help="Skip creating audio_vad.csv")
    parser.add_argument("--skip_sliding_window", action="store_true", help="Skip creating sliding window dataset")
    parser.add_argument("--skip_data_check", action="store_true", help="Skip checking data with DataModule")
    parser.add_argument("--skip_event_extraction", action="store_true", help="Skip extracting event dataset")
    
    return parser.parse_args()


def create_directory_structure(output_dir):
    """Create necessary directories for output files"""
    paths = {
        "main": Path(output_dir),
        "splits": Path(output_dir) / "splits",
        "classification": Path(output_dir) / "classification"
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def split_dataset(input_csv, train_csv, val_csv, test_csv, val_size=0.15, test_size=0.15, seed=42):
    """Split a dataset into training, validation and test sets"""
    df = pd.read_csv(input_csv)
    
    # Get unique sessions to prevent session leakage between splits
    sessions = df['session'].unique()
    
    # First split off the test set
    train_val_sessions, test_sessions = train_test_split(sessions, test_size=test_size, random_state=seed)
    
    # Then split the remaining data into train and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to the remaining data
    train_sessions, val_sessions = train_test_split(train_val_sessions, test_size=val_size_adjusted, random_state=seed)
    
    # Filter dataframes based on sessions
    train_df = df[df['session'].isin(train_sessions)]
    val_df = df[df['session'].isin(val_sessions)]
    test_df = df[df['session'].isin(test_sessions)]
    
    # Save to CSV
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Split dataset: {len(train_df)} training samples, {len(val_df)} validation samples, {len(test_df)} test samples")
    return train_csv, val_csv, test_csv


def check_data_with_datamodule(train_csv, val_csv, batch_size=4, num_workers=0):
    """Check if data can be properly loaded with DataModule"""
    dm = VAPDataModule(
        train_path=train_csv,
        val_path=val_csv,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    print("Setting up DataModule...")
    dm.prepare_data()
    dm.setup("fit")
    
    print("DataModule configuration:")
    print(dm)
    
    print("Checking training dataloader...")
    train_batch = next(iter(dm.train_dataloader()))
    print(f"Training batch shapes: waveform={train_batch['waveform'].shape}, vad={train_batch['vad'].shape}")
    
    print("Checking validation dataloader...")
    val_batch = next(iter(dm.val_dataloader()))
    print(f"Validation batch shapes: waveform={val_batch['waveform'].shape}, vad={val_batch['vad'].shape}")
    
    print("Data check completed successfully!")


def get_test_sessions(test_csv):
    """Extract session information from test dataset"""
    df = pd.read_csv(test_csv)
    return set(df['session'].unique())


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create directory structure
    paths = create_directory_structure(args.output_dir)
    
    # Define file paths
    audio_vad_csv = paths["main"] / "audio_vad.csv"
    sliding_window_csv = paths["main"] / "sliding_window_dset.csv"
    train_csv = paths["splits"] / "sliding_window_dset_train.csv"
    val_csv = paths["splits"] / "sliding_window_dset_val.csv"
    test_csv = paths["splits"] / "sliding_window_dset_test.csv"
    event_csv = paths["classification"] / "event_all.csv"
    test_event_csv = paths["classification"] / "test_dset_event.csv"
    
    # Step 1: Create audio_vad.csv
    if not args.skip_audio_vad_csv:
        print("\n===== Step 1: Creating audio_vad.csv =====")
        # Create a namespace object to match the expected arguments for create_audio_vad_csv
        audio_vad_args = argparse.Namespace(
            audio_dir=args.audio_dir,
            vad_dir=args.vad_dir,
            output=str(audio_vad_csv)
        )
        create_audio_vad_csv(audio_vad_args)
    else:
        print("\n===== Step 1: Skipping audio_vad.csv creation =====")
    
    # Step 2: Create sliding window dataset
    if not args.skip_sliding_window and os.path.exists(audio_vad_csv):
        print("\n===== Step 2: Creating sliding window dataset =====")
        # Create a namespace object for create_sliding_window_dset
        sliding_window_args = argparse.Namespace(
            audio_vad_csv=str(audio_vad_csv),
            output=str(sliding_window_csv),
            duration=args.duration,
            overlap=args.overlap,
            horizon=args.horizon
        )
        sliding_window_module.main(sliding_window_args)
        
        # Split dataset into train, validation, and test sets
        print("\n===== Step 3: Splitting dataset into train, validation, and test sets =====")
        split_dataset(sliding_window_csv, train_csv, val_csv, test_csv, args.val_size, args.test_size, args.seed)
    else:
        print("\n===== Step 2 & 3: Skipping sliding window dataset creation and splitting =====")
    
    # Step 4: Check if data can be properly loaded with DataModule
    if not args.skip_data_check and os.path.exists(train_csv) and os.path.exists(val_csv):
        print("\n===== Step 4: Checking data with DataModule =====")
        try:
            check_data_with_datamodule(train_csv, val_csv, args.batch_size, args.num_workers)
        except Exception as e:
            print(f"Error checking data: {e}")
    else:
        print("\n===== Step 4: Skipping data check =====")
    
    # Step 5: Extract event datasets (for evaluation)
    if not args.skip_event_extraction and os.path.exists(audio_vad_csv):
        print("\n===== Step 5: Extracting event datasets for evaluation =====")
        # Extract events for all data
        print("Creating complete event dataset...")
        create_classification_dset(
            audio_vad_path=str(audio_vad_csv),
            output=str(event_csv),
            pre_cond_time=args.pre_cond_time,
            post_cond_time=args.post_cond_time,
            min_silence_time=args.min_silence_time,
            ipu_based_events=args.ipu_based_events
        )
        
        # Extract events specifically for test data
        if os.path.exists(test_csv):
            print("Creating test-specific event dataset...")
            # Get test sessions
            test_sessions = get_test_sessions(test_csv)
            
            # Read the complete event dataset
            all_events_df = pd.read_csv(event_csv)
            
            # Extract audio paths from events dataframe
            audio_paths = all_events_df['audio_path'].unique()
            
            # Filter for test sessions
            test_events = []
            for audio_path in audio_paths:
                session = Path(audio_path).stem
                if session in test_sessions:
                    session_events = all_events_df[all_events_df['audio_path'] == audio_path]
                    test_events.append(session_events)
            
            if test_events:
                # Combine all test events
                test_events_df = pd.concat(test_events, ignore_index=True)
                test_events_df.to_csv(test_event_csv, index=False)
                print(f"Saved {len(test_events_df)} test events to {test_event_csv}")
            else:
                print("No test events found!")
    else:
        print("\n===== Step 5: Skipping event dataset extraction =====")
    
    print("\n===== Workflow completed! =====")
    print(f"Audio-VAD CSV: {audio_vad_csv}")
    print(f"Sliding window dataset: {sliding_window_csv}")
    print(f"Training dataset: {train_csv}")
    print(f"Validation dataset: {val_csv}")
    print(f"Test dataset: {test_csv}")
    print(f"Complete event dataset: {event_csv}")
    print(f"Test event dataset: {test_event_csv}")


if __name__ == "__main__":
    main()
