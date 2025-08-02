import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
import os
import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Mapping
from PIL import Image
import glob

from vap.utils.audio import load_waveform, mono_to_stereo
from vap.utils.utils import vad_list_to_onehot
from vap.data.datamodule import VAPDataset, VAPDataModule, force_correct_nsamples


SAMPLE = Mapping[str, torch.Tensor]


def load_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image from the given path
    """
    if not os.path.exists(image_path):
        # Return a blank image if file doesn't exist
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Load image
    img = cv2.imread(image_path)
    
    if img is None:
        # Return a blank image if loading failed
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed
    if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
        img = cv2.resize(img, target_size)
    
    return img


def get_image_paths_for_window(image_dir, session, start_time, end_time, video_fps=30, speaker_type="customer"):
    """
    Get paths to image files for a specific time window
    
    Args:
        image_dir: Base directory for images
        session: Session ID
        start_time: Start time in seconds
        end_time: End time in seconds
        video_fps: Video frames per second
        speaker_type: "customer" or "operator"
        
    Returns:
        List of image file paths
    """
    if not image_dir:
        return []
    
    # Calculate frame range based on time window
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    
    # Pattern for matching speaker type
    prefix = "customer_" if speaker_type == "customer" else "operator_"
    
    # Get all matching image files
    pattern = os.path.join(image_dir, f"{prefix}*.png")
    all_image_files = glob.glob(pattern)
    
    if not all_image_files:
        return []
    
    # Extract frame numbers and sort by them
    image_paths = []
    for frame_idx in range(start_frame, end_frame + 1):
        # Format the frame number with leading zeros (assuming 6 digits)
        frame_str = f"{frame_idx:06d}"
        frame_path = os.path.join(image_dir, f"{prefix}{frame_str}.png")
        
        if os.path.exists(frame_path):
            image_paths.append(frame_path)
        else:
            # If specific frame doesn't exist, use a placeholder
            image_paths.append(None)
    
    return image_paths


class MultimodalVAPDataset(Dataset):
    def __init__(
        self,
        path: str,
        horizon: float = 2,
        duration: float = 20,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        video_fps: int = 30,
        target_image_size: Tuple[int, int] = (224, 224),
        use_images: bool = True,
        eval_mode: bool = False,
        eval_single_speaker: Optional[int] = None,
    ) -> None:
        """
        Multimodal VAP Dataset that handles both audio and visual data
        
        Args:
            path: Path to CSV file with dataset information
            horizon: Prediction horizon in seconds
            duration: Duration of audio clips in seconds
            sample_rate: Audio sample rate
            frame_hz: Frame rate for audio features
            video_fps: Frame rate for video
            target_image_size: Target size for images
            use_images: Whether to use images or audio only
            eval_mode: Whether to use evaluation mode
            eval_single_speaker: Which speaker to use in evaluation mode (0 or 1)
        """
        self.path = path
        self.df = self._load_df(path)
        
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.video_fps = video_fps
        self.target_image_size = target_image_size
        self.horizon = horizon
        self.use_images = use_images
        self.eval_mode = eval_mode
        self.eval_single_speaker = eval_single_speaker
        
        self.duration = duration
        self.n_samples = int(self.duration * self.sample_rate)
        
        # Number of frames for the video
        self.n_video_frames = int(self.duration * self.video_fps)
        
        print(f"Loaded dataset from {path} with {len(self.df)} samples")
        print(f"Using images: {use_images}")
        if eval_mode:
            print(f"Eval mode: True, single speaker: {eval_single_speaker}")
            
    def _load_df(self, path: str) -> pd.DataFrame:
        """
        Load the dataset CSV file
        
        Handles two different CSV formats:
        - Training/validation: session,audio_path,start,end,vad_list,image_dir,has_images
        - Test: ipu_end,tfo,speaker,label,audio_path,vad_path,session,image_dir,has_images
        """
        def _vl(x):
            if pd.isna(x):
                return []
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                print(f"Error parsing JSON: {x}")
                return []

        def _session(x):
            return str(x)

        converters = {
            "vad_list": _vl,
            "session": _session,
        }
        
        df = pd.read_csv(path, converters=converters)
        
        # Print dataset format information for debugging
        print(f"CSV columns: {df.columns.tolist()}")
        
        # Check if this is the test dataset format
        is_test_format = "ipu_end" in df.columns and "start" not in df.columns
        
        if is_test_format:
            print(f"Loaded test format CSV with {len(df)} samples")
        else:
            print(f"Loaded train/val format CSV with {len(df)} samples")
            
        return df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset
        """
        d = self.df.iloc[idx]
        
        # Handle different column structures for train/val vs test datasets
        # Training/validation CSVs have: session,audio_path,start,end,vad_list,image_dir,has_images
        # Test CSV has: ipu_end,tfo,speaker,label,audio_path,vad_path,session,image_dir,has_images
        
        # Get start and end times based on available columns
        if "start" in d and "end" in d:
            # Training/validation CSV format
            start_time = d["start"]
            end_time = d["end"]
            # Duration can be 19.99999999999997 for some clips and result in wrong vad-shape
            # so we round it to nearest second
            dur = round(end_time - start_time)
        elif "ipu_end" in d:
            # Test CSV format - calculate start and end times
            # For test data, assume fixed duration and use ipu_end as reference
            end_time = d["ipu_end"]
            start_time = end_time - self.duration
            dur = self.duration
            
        # Load audio waveform
        w, _ = load_waveform(
            d["audio_path"],
            start_time=start_time,
            end_time=end_time,
            sample_rate=self.sample_rate,
            mono=False,
        )
        
        # Ensure correct duration for audio
        w = force_correct_nsamples(w, self.n_samples)
        
        # Handle VAD list based on available columns
        if "vad_list" in d:
            # Training/validation CSV has vad_list directly
            vad_list = d["vad_list"]
        elif "vad_path" in d:
            # Test CSV has vad_path pointing to a file with VAD data
            try:
                with open(d["vad_path"], 'r') as f:
                    vad_list = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading VAD data from {d['vad_path']}: {e}")
                # Provide empty VAD list as fallback
                vad_list = []
        else:
            # Fallback if no VAD data available
            vad_list = []
            
        # Stereo Audio - Use VAD list to convert mono to stereo if needed
        if w.shape[0] == 1:
            w = mono_to_stereo(w, vad_list, self.sample_rate)
        
        # Create VAD tensor
        vad = vad_list_to_onehot(
            vad_list, duration=dur + self.horizon, frame_hz=self.frame_hz
        )
        
        # Prepare the basic sample with audio and VAD
        sample = {
            "session": d.get("session", ""),
            "waveform": w,
            "vad": vad,
            "dataset": d.get("dataset", ""),
        }
        
        # If we're using images and image directory is available
        if self.use_images and d.get("has_images", False) and d.get("image_dir", ""):
            image_dir = d["image_dir"]
            session = d["session"]
            # Using start_time and end_time from previous calculations
            # which handles both train/val and test dataset formats
            
            # Get image paths for both speakers
            customer_paths = get_image_paths_for_window(
                image_dir, session, start_time, end_time, 
                self.video_fps, "customer"
            )
            operator_paths = get_image_paths_for_window(
                image_dir, session, start_time, end_time, 
                self.video_fps, "operator"
            )
            
            # If we have enough images for both speakers
            has_enough_images = (len(customer_paths) >= self.n_video_frames * 0.5 and 
                                len(operator_paths) >= self.n_video_frames * 0.5)
            
            if has_enough_images:
                # Load images for both speakers
                speaker1_images = []
                speaker2_images = []
                
                # Determine which speaker is which based on the session
                # In this implementation, we assume customer is speaker 1 and operator is speaker 2
                speaker1_paths = customer_paths
                speaker2_paths = operator_paths
                
                # Process as many frames as needed for the duration
                for i in range(min(self.n_video_frames, len(speaker1_paths))):
                    if i < len(speaker1_paths) and speaker1_paths[i]:
                        img = load_image(speaker1_paths[i], self.target_image_size)
                    else:
                        img = np.zeros((*self.target_image_size, 3), dtype=np.uint8)
                    speaker1_images.append(img)
                    
                for i in range(min(self.n_video_frames, len(speaker2_paths))):
                    if i < len(speaker2_paths) and speaker2_paths[i]:
                        img = load_image(speaker2_paths[i], self.target_image_size)
                    else:
                        img = np.zeros((*self.target_image_size, 3), dtype=np.uint8)
                    speaker2_images.append(img)
                
                # Convert to numpy arrays and normalize
                speaker1_images = np.array(speaker1_images).astype(np.float32) / 255.0
                speaker2_images = np.array(speaker2_images).astype(np.float32) / 255.0
                
                # Convert to torch tensors and adjust dimensions
                # From [F, H, W, C] to [F, C, H, W]
                speaker1_images = torch.tensor(speaker1_images).permute(0, 3, 1, 2)
                speaker2_images = torch.tensor(speaker2_images).permute(0, 3, 1, 2)
                
                # Add to sample
                sample["images1"] = speaker1_images
                sample["images2"] = speaker2_images
                sample["has_images"] = True
            else:
                # Not enough images, return empty tensors
                sample["images1"] = torch.zeros(1, 3, *self.target_image_size)
                sample["images2"] = torch.zeros(1, 3, *self.target_image_size)
                sample["has_images"] = False
        else:
            # No images available
            sample["images1"] = torch.zeros(1, 3, *self.target_image_size)
            sample["images2"] = torch.zeros(1, 3, *self.target_image_size)
            sample["has_images"] = False
        
        # For evaluation mode with single speaker
        if self.eval_mode and self.eval_single_speaker is not None:
            # Create a copy of the audio data where we only provide one speaker's audio
            eval_waveform = w.clone()
            if self.eval_single_speaker == 0:
                # Zero out speaker 2
                eval_waveform[1, :] = 0
            else:
                # Zero out speaker 1
                eval_waveform[0, :] = 0
            
            sample["eval_waveform"] = eval_waveform
        
        return sample


class MultimodalVAPDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        horizon: float = 2,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        video_fps: int = 30,
        target_image_size: Tuple[int, int] = (224, 224),
        use_images: bool = True,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        eval_mode: bool = False,
        **kwargs,
    ):
        """
        DataModule for multimodal VAP model
        """
        super().__init__()

        # Files
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # Dataset parameters
        self.horizon = horizon
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.video_fps = video_fps
        self.target_image_size = target_image_size
        self.use_images = use_images
        self.eval_mode = eval_mode

        # DataLoader parameters
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tHorizon: {self.horizon}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += f"\n\tVideo FPS: {self.video_fps}"
        s += f"\n\tUse Images: {self.use_images}"
        s += f"\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        s += f"\n\tprefetch_factor: {self.prefetch_factor}"
        return s

    def prepare_data(self):
        """
        Check if data files exist
        """
        if self.train_path is not None:
            assert os.path.isfile(self.train_path), f"Train file not found: {self.train_path}"

        if self.val_path is not None:
            assert os.path.isfile(self.val_path), f"Val file not found: {self.val_path}"

        if self.test_path is not None:
            assert os.path.isfile(self.test_path), f"Test file not found: {self.test_path}"

    def setup(self, stage: Optional[str] = "fit"):
        """
        Setup datasets for the different stages
        """
        if stage in (None, "fit"):
            if self.train_path is not None:
                self.train_dset = MultimodalVAPDataset(
                    path=self.train_path,
                    horizon=self.horizon,
                    sample_rate=self.sample_rate,
                    frame_hz=self.frame_hz,
                    video_fps=self.video_fps,
                    target_image_size=self.target_image_size,
                    use_images=self.use_images,
                    eval_mode=False,  # Training doesn't use eval mode
                )

            if self.val_path is not None:
                self.val_dset = MultimodalVAPDataset(
                    path=self.val_path,
                    horizon=self.horizon,
                    sample_rate=self.sample_rate,
                    frame_hz=self.frame_hz,
                    video_fps=self.video_fps,
                    target_image_size=self.target_image_size,
                    use_images=self.use_images,
                    eval_mode=False,  # Validation doesn't use eval mode either
                )

        if stage in (None, "test"):
            if self.test_path is not None:
                # For testing, we create two datasets with different eval_single_speaker values
                self.test_dset = MultimodalVAPDataset(
                    path=self.test_path,
                    horizon=self.horizon,
                    sample_rate=self.sample_rate,
                    frame_hz=self.frame_hz,
                    video_fps=self.video_fps,
                    target_image_size=self.target_image_size,
                    use_images=self.use_images,
                    eval_mode=self.eval_mode,
                    eval_single_speaker=None,  # Normal mode with both speakers
                )
                
                if self.eval_mode:
                    # Create additional datasets for single-speaker evaluation
                    self.test_dset_speaker0 = MultimodalVAPDataset(
                        path=self.test_path,
                        horizon=self.horizon,
                        sample_rate=self.sample_rate,
                        frame_hz=self.frame_hz,
                        video_fps=self.video_fps,
                        target_image_size=self.target_image_size,
                        use_images=self.use_images,
                        eval_mode=True,
                        eval_single_speaker=0,  # Only speaker 0
                    )
                    
                    self.test_dset_speaker1 = MultimodalVAPDataset(
                        path=self.test_path,
                        horizon=self.horizon,
                        sample_rate=self.sample_rate,
                        frame_hz=self.frame_hz,
                        video_fps=self.video_fps,
                        target_image_size=self.target_image_size,
                        use_images=self.use_images,
                        eval_mode=True,
                        eval_single_speaker=1,  # Only speaker 1
                    )

    def collate_fn(self, batch: List[Dict[str, Any]]):
        """
        Custom collate function to handle variable-sized data
        """
        batch_stacked = {k: [] for k in batch[0].keys()}

        for b in batch:
            for k, v in b.items():
                batch_stacked[k].append(v)

        # Stack tensors
        batch_stacked["waveform"] = torch.stack(batch_stacked["waveform"])
        batch_stacked["vad"] = torch.stack(batch_stacked["vad"])
        
        # Stack images if they have consistent sizes
        if all(img.shape[0] == batch_stacked["images1"][0].shape[0] for img in batch_stacked["images1"]):
            batch_stacked["images1"] = torch.stack(batch_stacked["images1"])
            batch_stacked["images2"] = torch.stack(batch_stacked["images2"])
        else:
            # Handle variable-length frame sequences by padding to the max length
            max_frames = max(img.shape[0] for img in batch_stacked["images1"])
            
            # Pad each sequence to the max length
            padded_images1 = []
            padded_images2 = []
            
            for img1, img2 in zip(batch_stacked["images1"], batch_stacked["images2"]):
                # Get current frames and channels
                frames1, channels1, height1, width1 = img1.shape
                frames2, channels2, height2, width2 = img2.shape
                
                # Create padded tensors
                padded1 = torch.zeros(max_frames, channels1, height1, width1)
                padded2 = torch.zeros(max_frames, channels2, height2, width2)
                
                # Copy data
                padded1[:frames1] = img1
                padded2[:frames2] = img2
                
                # Add to lists
                padded_images1.append(padded1)
                padded_images2.append(padded2)
            
            # Stack padded sequences
            batch_stacked["images1"] = torch.stack(padded_images1)
            batch_stacked["images2"] = torch.stack(padded_images2)
        
        # Handle eval_waveform if present
        if "eval_waveform" in batch_stacked:
            batch_stacked["eval_waveform"] = torch.stack(batch_stacked["eval_waveform"])

        return batch_stacked

    def train_dataloader(self):
        """
        Create training dataloader
        """
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Create validation dataloader
        """
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Create test dataloader(s)
        
        Returns either a single dataloader or a list of dataloaders for different evaluation modes
        """
        if self.eval_mode:
            # Return three test dataloaders for different evaluation scenarios
            return [
                DataLoader(
                    self.test_dset,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                ),
                DataLoader(
                    self.test_dset_speaker0,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                ),
                DataLoader(
                    self.test_dset_speaker1,
                    batch_size=self.batch_size,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor,
                    collate_fn=self.collate_fn,
                    shuffle=False,
                ),
            ]
        else:
            # Return just one test dataloader
            return DataLoader(
                self.test_dset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.collate_fn,
                shuffle=False,
            )
