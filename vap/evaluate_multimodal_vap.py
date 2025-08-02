#!/usr/bin/env python

import torch
import logging
import hydra
import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from typing import Dict, List, Optional, Tuple, Any, Iterable
import torch.nn.functional as F
from torchmetrics.functional.classification.accuracy import accuracy
from torchmetrics.functional import f1_score

log: logging.Logger = logging.getLogger(__name__)

def get_dialog_states(vad: torch.Tensor) -> torch.Tensor:
    """
    Convert VAD to dialog states
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1

def extract_backchannel_regions(
    vad: torch.Tensor, 
    ds: torch.Tensor,
    pre_cond_frames: int,
    post_cond_frames: int, 
    prediction_region_frames: int,
    min_context_frames: int,
    max_bc_frames: int,
    max_frame: int
):
    """
    Extract backchannel regions from VAD data
    (Simplified version from events.py)
    """
    from vap.utils.utils import find_island_idx_len
    from vap.events.events import fill_pauses
    
    backchannel = []
    pred_backchannel = []
    pred_backchannel_neg = []
    
    filled_vad = fill_pauses(vad, ds)
    
    for speaker in [0, 1]:
        start_of, duration_of, states = find_island_idx_len(filled_vad[..., speaker])
        if len(states) < 3:
            continue
            
        # Find sequences [0, 1, 0] (silence-speech-silence)
        triad_bc = torch.tensor([0, 1, 0]).to(vad.device)
        triads = states.unfold(0, size=3, step=1)
        steps = torch.where(
            (triads == triad_bc.unsqueeze(0)).sum(-1) == 3
        )[0]
        
        if len(steps) == 0:
            continue
            
        for pre_silence in steps:
            bc = pre_silence + 1
            post_silence = pre_silence + 2
            
            # Apply conditions (simplified for clarity)
            if (start_of[bc] < min_context_frames or
                start_of[bc] >= max_frame or
                duration_of[bc] > max_bc_frames or
                duration_of[pre_silence] < pre_cond_frames or
                duration_of[post_silence] < post_cond_frames):
                continue
                
            # Valid backchannel
            backchannel.append(
                (start_of[bc].item(), start_of[post_silence].item(), speaker)
            )
            
            # Prediction region
            pred_bc_start = start_of[bc] - prediction_region_frames
            if pred_bc_start < min_context_frames:
                continue
                
            pred_backchannel.append(
                (pred_bc_start.item(), start_of[bc].item(), speaker)
            )
            
            # For negative samples, find a different region
            offset = prediction_region_frames
            if start_of[bc] + offset < max_frame:
                pred_backchannel_neg.append(
                    (start_of[bc].item(), (start_of[bc] + offset).item(), speaker)
                )
    
    return {
        "backchannel": backchannel, 
        "pred_backchannel": pred_backchannel,
        "pred_backchannel_neg": pred_backchannel_neg
    }

@hydra.main(version_base=None, config_path="conf", config_name="multimodal_vap_eval_config")
def main(cfg: DictConfig) -> None:
    """
    Script for evaluating the Multimodal VAP model with detailed analysis
    
    Args:
        cfg: Configuration object from Hydra
    """
    # Make sure we're in evaluation mode
    cfg.eval_only = True
    cfg.datamodule.eval_mode = True
    
    # Set random seed
    seed = cfg.get("seed", 0)
    seed_everything(seed, workers=True)
    
    # Log configuration
    log.info(OmegaConf.to_yaml(cfg))
    
    # Check for checkpoint path
    if not getattr(cfg, "pretrained_checkpoint_path", None):
        log.error("No checkpoint path provided. Please specify pretrained_checkpoint_path.")
        return
    
    # Set output directory
    output_dir = cfg.get("output_dir", os.path.join(os.getcwd(), "eval_outputs"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Instantiate model and datamodule
    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    
    # Load from checkpoint
    module = module.load_from_checkpoint(
        checkpoint_path=cfg.pretrained_checkpoint_path,
        model=module.model,
        strict=False
    )
    log.info(f"Loaded from checkpoint: {cfg.pretrained_checkpoint_path}")
    
    # Add test metrics if needed
    if getattr(cfg.module, "test_metric", False):
        try:
            log.info("Adding test metrics...")
            test_metric = instantiate(cfg.module.test_metric)
            module.test_metric = test_metric
            log.info(f"Successfully added test metrics: {type(module.test_metric).__name__}")
        except Exception as e:
            log.error(f"Failed to instantiate test_metric: {str(e)}")
            module.test_metric = None
            log.warning("Setting test_metric to None")
    
    # Prepare data
    datamodule.prepare_data()
    datamodule.setup("test")
    
    # Ensure model is in evaluation mode and on correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module.to(device)
    module.eval()
    
    # Get test dataloaders
    test_dataloaders = datamodule.test_dataloader()
    if not isinstance(test_dataloaders, list):
        test_dataloaders = [test_dataloaders]
    
    # Define evaluation scenarios
    scenarios = ["Both speakers", "Speaker 0 only", "Speaker 1 only"]
    
    # Create results dictionary
    results = {scenario: {} for scenario in scenarios}
    
    # Settings for backchannel evaluation
    prediction_region_time = 0.5  # seconds
    frame_hz = 50  # Hz
    prediction_region_frames = int(prediction_region_time * frame_hz)
    pre_cond_frames = int(1.0 * frame_hz)
    post_cond_frames = int(1.0 * frame_hz)
    min_context_frames = int(3.0 * frame_hz)
    max_bc_frames = int(1.0 * frame_hz)
    max_frame = int(20.0 * frame_hz)
    threshold = 0.5
    
    # Evaluate each dataloader (different scenarios)
    for i, dataloader in enumerate(test_dataloaders):
        scenario = scenarios[min(i, len(scenarios) - 1)]
        log.info(f"Evaluating scenario: {scenario}")
        
        # Predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        all_speaker_info = []
        
        # Backchannel-specific data
        backchannel_regions = []
        backchannel_probs = []
        backchannel_preds = []
        backchannel_targets = []
        
        # Speaker-specific metrics
        speaker0_metrics = {"hs": [], "ls": [], "sp": [], "bp": []}
        speaker1_metrics = {"hs": [], "ls": [], "sp": [], "bp": []}
        
        # Process batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {scenario}"):
                # Move data to device
                waveform = batch["waveform"].to(device)
                vad = batch["vad"].to(device)
                
                # Get images if available and enabled
                images1 = batch.get("images1", None)
                images2 = batch.get("images2", None)
                
                if images1 is not None and images2 is not None and module.use_images:
                    images1 = images1.to(device)
                    images2 = images2.to(device)
                else:
                    images1, images2 = None, None
                
                # Override waveform for single-speaker scenarios if needed
                if i > 0 and "eval_waveform" in batch:
                    waveform = batch["eval_waveform"].to(device)
                
                # Forward pass
                out = module.model(waveform, images1, images2)
                
                # Get predictions
                logits = out["logits"]
                probs = F.softmax(logits, dim=-1)
                
                # Process predictions for shift vs hold events
                p_now = probs[:, :, 0:2].sum(dim=-1)  # Now region (first two bins)
                p_future = probs[:, :, 2:4].sum(dim=-1)  # Future region (last two bins)
                
                # Extract dialog states
                ds = get_dialog_states(vad)
                
                # Process each sample in batch
                for b in range(vad.shape[0]):
                    # Extract backchannel regions
                    bc_regions = extract_backchannel_regions(
                        vad[b],
                        ds[b],
                        pre_cond_frames=pre_cond_frames,
                        post_cond_frames=post_cond_frames,
                        prediction_region_frames=prediction_region_frames,
                        min_context_frames=min_context_frames,
                        max_bc_frames=max_bc_frames,
                        max_frame=max_frame
                    )
                    backchannel_regions.append(bc_regions)
                    
                    # Process backchannel predictions
                    sample_bc_probs = {}
                    
                    # Process positive samples (backchannel)
                    for start, end, speaker in bc_regions["pred_backchannel"]:
                        # Probability for backchannel
                        pbc = p_future[b, start:end].mean()
                        if speaker:
                            pbc = 1 - pbc  # Speaker adjustment
                        
                        backchannel_preds.append(pbc.cpu())
                        backchannel_targets.append(torch.ones(1))
                        sample_bc_probs[(start, end, speaker)] = pbc.item()
                        
                        # Add to speaker-specific metrics
                        if speaker == 0:
                            speaker0_metrics["bp"].append((pbc.cpu(), torch.ones(1)))
                        else:
                            speaker1_metrics["bp"].append((pbc.cpu(), torch.ones(1)))
                    
                    # Process negative samples (no backchannel)
                    for start, end, speaker in bc_regions["pred_backchannel_neg"]:
                        # Different region for negative samples
                        if start >= vad.shape[1] or end >= vad.shape[1]:
                            continue
                        
                        pbc_neg = 1 - p_future[b, start:end].mean()
                        if speaker:
                            pbc_neg = 1 - pbc_neg
                        
                        backchannel_preds.append(pbc_neg.cpu())
                        backchannel_targets.append(torch.zeros(1))
                        
                        # Add to speaker-specific metrics
                        if speaker == 0:
                            speaker0_metrics["bp"].append((pbc_neg.cpu(), torch.zeros(1)))
                        else:
                            speaker1_metrics["bp"].append((pbc_neg.cpu(), torch.zeros(1)))
                    
                    backchannel_probs.append(sample_bc_probs)
                    
                    # Process other events for speaker-specific metrics
                    # This is a simplified approach - in practice, you'd use the event extractor from metrics.py
                    # For demonstration purposes, we'll just separate by VAD
                    for t in range(vad.shape[1] - 10):  # Skip last few frames
                        # Skip if both speakers are silent or both are speaking
                        if vad[b, t, 0] == vad[b, t, 1]:
                            continue
                        
                        # Determine active speaker
                        speaker = 1 if vad[b, t, 1] > vad[b, t, 0] else 0
                        
                        # Get predictions at this time point
                        p_shift = p_future[b, t]
                        
                        # Add to speaker-specific metrics (simplified)
                        if speaker == 0:
                            # Shift vs Hold (hs)
                            speaker0_metrics["hs"].append((p_shift, torch.zeros(1) if t + 10 < vad.shape[1] and vad[b, t+10, 0] > 0 else torch.ones(1)))
                            # Long vs Short (ls)
                            speaker0_metrics["ls"].append((p_shift, torch.zeros(1) if t + 20 < vad.shape[1] and vad[b, t+20, 0] > 0 else torch.ones(1)))
                            # Shift Prediction (sp)
                            speaker0_metrics["sp"].append((p_shift, torch.ones(1) if t + 10 < vad.shape[1] and vad[b, t+10, 0] == 0 else torch.zeros(1)))
                        else:
                            # Same for speaker 1
                            speaker1_metrics["hs"].append((p_shift, torch.zeros(1) if t + 10 < vad.shape[1] and vad[b, t+10, 1] > 0 else torch.ones(1)))
                            speaker1_metrics["ls"].append((p_shift, torch.zeros(1) if t + 20 < vad.shape[1] and vad[b, t+20, 1] > 0 else torch.ones(1)))
                            speaker1_metrics["sp"].append((p_shift, torch.ones(1) if t + 10 < vad.shape[1] and vad[b, t+10, 1] == 0 else torch.zeros(1)))
                
                # Store predictions and ground truth
                all_predictions.append(probs.cpu())
                all_ground_truth.append(vad.cpu())
                
                # Store speaker info
                for session in batch["session"]:
                    all_speaker_info.append(session)
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        
        # Calculate standard metrics
        if hasattr(module, "test_metric") and module.test_metric is not None:
            try:
                module.test_metric.reset()
                module.test_metric.update_batch(all_predictions, all_ground_truth)
                metrics = module.test_metric.compute()
                
                # Store metrics in results
                results[scenario]["metrics"] = metrics
                
                # Print metrics
                log.info(f"Metrics for {scenario}:")
                for event_name, metric in metrics.items():
                    log.info(f"  {event_name}: F1={metric['f1']:.4f}, ACC0={metric['acc'][0]:.4f}, ACC1={metric['acc'][1]:.4f}")
            except Exception as e:
                log.error(f"Error computing metrics: {str(e)}")
                results[scenario]["metrics"] = {"error": str(e)}
                log.warning(f"Failed to compute metrics for {scenario}")
        else:
            log.warning(f"No test_metric available for {scenario}, skipping metrics computation")
        
        # Calculate backchannel metrics
        bc_metrics = {}
        if backchannel_preds and backchannel_targets:
            bc_preds_tensor = torch.cat(backchannel_preds)
            bc_targets_tensor = torch.cat(backchannel_targets)
            
            # Binary predictions
            bc_binary_preds = (bc_preds_tensor >= threshold).float()
            
            # Calculate accuracy by class
            bc_acc = accuracy(
                bc_binary_preds,
                bc_targets_tensor,
                task="multiclass",
                num_classes=2,
                average="none"
            )
            
            # Calculate F1 score
            bc_f1 = f1_score(
                bc_binary_preds,
                bc_targets_tensor,
                task="multiclass",
                num_classes=2,
                average="weighted"
            )
            
            bc_metrics = {
                "bp": {
                    "acc": bc_acc,
                    "f1": bc_f1
                }
            }
            
            # Print backchannel metrics
            log.info(f"Backchannel Prediction for {scenario}:")
            log.info(f"  No-Backchannel Accuracy: {bc_acc[0]:.4f}")
            log.info(f"  Backchannel Accuracy: {bc_acc[1]:.4f}")
            log.info(f"  F1 Score: {bc_f1:.4f}")
            log.info(f"  Total samples: {len(bc_preds_tensor)}")
            
            # Add backchannel metrics to results
            if "metrics" not in results[scenario]:
                results[scenario]["metrics"] = {}
            results[scenario]["metrics"].update(bc_metrics)
        
        # Calculate speaker-specific metrics
        for speaker_idx, speaker_metrics in [(0, speaker0_metrics), (1, speaker1_metrics)]:
            log.info(f"Speaker {speaker_idx} metrics for {scenario}:")
            speaker_results = {}
            
            for event_type, metrics_data in speaker_metrics.items():
                if not metrics_data:
                    continue
                
                # Combine predictions and targets
                preds, targets = zip(*metrics_data)
                preds_tensor = torch.cat([p.unsqueeze(0) for p in preds])
                targets_tensor = torch.cat([t.unsqueeze(0) for t in targets])
                
                # Binary predictions
                binary_preds = (preds_tensor >= threshold).float()
                
                # Calculate accuracy by class
                event_acc = accuracy(
                    binary_preds,
                    targets_tensor,
                    task="multiclass",
                    num_classes=2,
                    average="none"
                )
                
                # Calculate F1 score
                event_f1 = f1_score(
                    binary_preds,
                    targets_tensor,
                    task="multiclass",
                    num_classes=2,
                    average="weighted"
                )
                
                speaker_results[f"{event_type}_speaker{speaker_idx}"] = {
                    "acc": event_acc,
                    "f1": event_f1
                }
                
                # Print metrics
                log.info(f"  {event_type}: F1={event_f1:.4f}, ACC0={event_acc[0]:.4f}, ACC1={event_acc[1]:.4f}")
            
            # Add to results
            if "speaker_metrics" not in results[scenario]:
                results[scenario]["speaker_metrics"] = {}
            results[scenario]["speaker_metrics"][f"speaker{speaker_idx}"] = speaker_results
        
        # Store predictions and ground truth
        results[scenario]["predictions"] = all_predictions
        results[scenario]["ground_truth"] = all_ground_truth
        results[scenario]["speaker_info"] = all_speaker_info
        results[scenario]["backchannel"] = {
            "regions": backchannel_regions,
            "probs": backchannel_probs,
            "metrics": bc_metrics
        }
        
        # Generate visualizations
        generate_visualizations(
            all_predictions, 
            all_ground_truth, 
            results[scenario]["backchannel"],
            os.path.join(output_dir, f"viz_{scenario.replace(' ', '_')}"),
            scenario
        )
    
    # Save results (excluding large tensors)
    metrics_only = {}
    for scenario in results:
        metrics_only[scenario] = {
            "metrics": results[scenario].get("metrics", {}),
            "speaker_metrics": results[scenario].get("speaker_metrics", {})
        }
    
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_only, f, indent=2)
    
    # Generate overall summary
    generate_summary_report(metrics_only, output_dir)
    
    log.info(f"Evaluation completed. Results saved to {output_dir}")


def generate_summary_report(metrics_data, output_dir):
    """
    Generate a summary report of all metrics across scenarios
    
    Args:
        metrics_data: Dictionary with metrics for all scenarios
        output_dir: Directory to save the report
    """
    # Create a comprehensive report
    from datetime import datetime
    report = {
        "timestamp": str(datetime.now()),
        "metrics_by_scenario": metrics_data
    }
    
    # Prepare data for visualization
    scenarios = list(metrics_data.keys())
    event_types = ["hs", "ls", "sp", "bp"]
    event_names = ["Hold vs Shift", "Long vs Short", "Shift Prediction", "Backchannel"]
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # 1. Overall F1 scores by scenario and event type
    plt.subplot(2, 2, 1)
    
    # Extract F1 scores
    f1_data = {}
    for scenario in scenarios:
        f1_data[scenario] = []
        for event in event_types:
            try:
                f1_data[scenario].append(metrics_data[scenario]["metrics"].get(event, {}).get("f1", 0))
            except (KeyError, AttributeError):
                f1_data[scenario].append(0)
    
    # Plot grouped bar chart
    width = 0.2
    x = np.arange(len(event_names))
    
    for i, scenario in enumerate(scenarios):
        offset = (i - len(scenarios) / 2 + 0.5) * width
        plt.bar(x + offset, f1_data[scenario], width, label=scenario)
    
    plt.title('F1 Scores by Scenario and Event Type')
    plt.ylabel('F1 Score')
    plt.xlabel('Event Type')
    plt.xticks(x, event_names)
    plt.ylim(0, 1)
    plt.legend()
    
    # 2. Speaker-specific metrics (Speaker 0)
    plt.subplot(2, 2, 2)
    speaker0_data = {}
    
    for scenario in scenarios:
        speaker0_data[scenario] = []
        try:
            speaker_metrics = metrics_data[scenario].get("speaker_metrics", {}).get("speaker0", {})
            for event in event_types:
                event_key = f"{event}_speaker0"
                speaker0_data[scenario].append(speaker_metrics.get(event_key, {}).get("f1", 0))
        except (KeyError, AttributeError):
            speaker0_data[scenario] = [0, 0, 0, 0]  # Defaults
    
    # Plot Speaker 0 data
    for i, scenario in enumerate(scenarios):
        offset = (i - len(scenarios) / 2 + 0.5) * width
        plt.bar(x + offset, speaker0_data[scenario], width, label=scenario)
    
    plt.title('Speaker 0 F1 Scores by Event Type')
    plt.ylabel('F1 Score')
    plt.xlabel('Event Type')
    plt.xticks(x, event_names)
    plt.ylim(0, 1)
    plt.legend()
    
    # 3. Speaker-specific metrics (Speaker 1)
    plt.subplot(2, 2, 3)
    speaker1_data = {}
    
    for scenario in scenarios:
        speaker1_data[scenario] = []
        try:
            speaker_metrics = metrics_data[scenario].get("speaker_metrics", {}).get("speaker1", {})
            for event in event_types:
                event_key = f"{event}_speaker1"
                speaker1_data[scenario].append(speaker_metrics.get(event_key, {}).get("f1", 0))
        except (KeyError, AttributeError):
            speaker1_data[scenario] = [0, 0, 0, 0]  # Defaults
    
    # Plot Speaker 1 data
    for i, scenario in enumerate(scenarios):
        offset = (i - len(scenarios) / 2 + 0.5) * width
        plt.bar(x + offset, speaker1_data[scenario], width, label=scenario)
    
    plt.title('Speaker 1 F1 Scores by Event Type')
    plt.ylabel('F1 Score')
    plt.xlabel('Event Type')
    plt.xticks(x, event_names)
    plt.ylim(0, 1)
    plt.legend()
    
    # 4. Comparison between speakers
    plt.subplot(2, 2, 4)
    
    # Calculate average F1 across scenarios
    speaker0_avg = [sum(speaker0_data[s][i] for s in scenarios) / len(scenarios) for i in range(len(event_types))]
    speaker1_avg = [sum(speaker1_data[s][i] for s in scenarios) / len(scenarios) for i in range(len(event_types))]
    
    # Plot comparison
    plt.bar(x - width/2, speaker0_avg, width, label='Speaker 0', color='blue')
    plt.bar(x + width/2, speaker1_avg, width, label='Speaker 1', color='red')
    
    plt.title('Speaker Comparison: Average F1 Scores')
    plt.ylabel('F1 Score')
    plt.xlabel('Event Type')
    plt.xticks(x, event_names)
    plt.ylim(0, 1)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_report.png'))
    plt.close()
    
    # Save detailed report as JSON
    with open(os.path.join(output_dir, "summary_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    log.info(f"Summary report generated and saved to {output_dir}")


def generate_visualizations(predictions, ground_truth, backchannel_data, output_dir, scenario=""):
    """
    Generate visualization plots for model predictions
    
    Args:
        predictions: Model predictions tensor
        ground_truth: Ground truth tensor
        backchannel_data: Dictionary with backchannel predictions and ground truth
        output_dir: Directory to save visualizations
        scenario: Evaluation scenario (e.g., "Both speakers", "Speaker 0 only")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create aggregated visualizations
    
    # 1. Sample-level visualizations (first few samples)
    num_samples = min(5, predictions.shape[0])
    
    for i in range(num_samples):
        plt.figure(figsize=(15, 12))
        
        # Plot ground truth VAD
        plt.subplot(4, 1, 1)
        vad = ground_truth[i].numpy()
        time_axis = np.arange(vad.shape[0]) / 50  # 50Hz
        plt.plot(time_axis, vad[:, 0], 'b-', label='Speaker 1')
        plt.plot(time_axis, vad[:, 1], 'r-', label='Speaker 2')
        plt.title(f'Ground Truth VAD - Sample {i+1} ({scenario})')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        
        # Plot prediction probabilities - Shift
        plt.subplot(4, 1, 2)
        probs = predictions[i].numpy()
        time_axis = np.arange(probs.shape[0]) / 50
        
        # Get the bin times from the model
        bin_times = [0.2, 0.4, 0.6, 0.8]  # Default bins
        
        # Plot each probability bin
        for bin_idx, bin_time in enumerate(bin_times):
            plt.plot(time_axis, probs[:, bin_idx], 
                    label=f'P(Shift in {bin_time}s)')
        
        plt.title(f'Shift Probabilities - Sample {i+1}')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        
        # Plot the binary predictions (thresholded) - Shift
        plt.subplot(4, 1, 3)
        threshold = 0.5
        binary_preds = (probs > threshold).astype(float)
        for bin_idx, bin_time in enumerate(bin_times):
            plt.plot(time_axis, binary_preds[:, bin_idx], 
                    label=f'Shift in {bin_time}s')
        
        plt.title(f'Binary Shift Predictions (threshold={threshold}) - Sample {i+1}')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        
        # Plot backchannel predictions if available
        plt.subplot(4, 1, 4)
        
        # Extract dialog states for visualization
        ds = get_dialog_states(torch.tensor(vad))
        
        # Highlight backchannel regions
        if i < len(backchannel_data.get("regions", [])):
            bc_regions = backchannel_data["regions"][i]
            
            # Plot dialog state as background
            plt.plot(time_axis, ds.cpu().numpy() / 4, 'k-', alpha=0.3, label='Dialog State')
            
            # Plot backchannel probabilities
            if "probs" in backchannel_data and i < len(backchannel_data["probs"]):
                bc_probs = backchannel_data["probs"][i]
                for start, end, speaker in bc_regions.get("pred_backchannel", []):
                    if start < len(time_axis) and end <= len(time_axis):
                        segment_time = time_axis[start:end]
                        # Get the backchannel probability for this segment
                        prob_values = np.zeros_like(time_axis)
                        prob_values[start:end] = bc_probs.get((start, end, speaker), 0.5)
                        plt.plot(segment_time, prob_values[start:end], 
                                'g-', linewidth=2, label=f'P(BC) Speaker {speaker}')
            
            # Highlight actual backchannel regions
            for start, end, speaker in bc_regions.get("backchannel", []):
                if start < len(time_axis) and end <= len(time_axis):
                    plt.axvspan(time_axis[start], time_axis[end], 
                               alpha=0.2, color='yellow', label=f'BC Speaker {speaker}')
        
        plt.title(f'Backchannel Regions - Sample {i+1}')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}_predictions.png'))
        plt.close()
    
    # 2. Aggregated metrics
    # Calculate average probabilities over time
    avg_probs = predictions.mean(dim=0).numpy()
    time_axis = np.arange(avg_probs.shape[0]) / 50
    
    plt.figure(figsize=(12, 6))
    bin_times = [0.2, 0.4, 0.6, 0.8]  # Default bins
    for bin_idx, bin_time in enumerate(bin_times):
        plt.plot(time_axis, avg_probs[:, bin_idx], 
                label=f'P(Shift in {bin_time}s)')
    
    plt.title(f'Average Shift Probabilities Over All Samples ({scenario})')
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'average_probabilities.png'))
    plt.close()
    
    # 3. Metrics visualization
    if "metrics" in backchannel_data:
        metrics = backchannel_data["metrics"]
        
        # Plot metrics for different event types
        event_types = ["hs", "ls", "sp", "bp"]
        event_names = ["Hold vs Shift", "Long vs Short", "Shift Prediction", "Backchannel Prediction"]
        
        plt.figure(figsize=(12, 8))
        
        # F1 scores
        plt.subplot(2, 1, 1)
        f1_scores = [metrics.get(event, {}).get("f1", 0) for event in event_types]
        plt.bar(event_names, f1_scores, color=['blue', 'green', 'red', 'purple'])
        plt.title(f'F1 Scores by Event Type ({scenario})')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1)
        
        # Accuracy by class
        plt.subplot(2, 1, 2)
        acc0_scores = [metrics.get(event, {}).get("acc", [0, 0])[0] for event in event_types]
        acc1_scores = [metrics.get(event, {}).get("acc", [0, 0])[1] for event in event_types]
        
        x = np.arange(len(event_names))
        width = 0.35
        
        plt.bar(x - width/2, acc0_scores, width, label='Class 0', color='lightskyblue')
        plt.bar(x + width/2, acc1_scores, width, label='Class 1', color='lightcoral')
        
        plt.xlabel('Event Type')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Class and Event Type ({scenario})')
        plt.xticks(x, event_names)
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_summary.png'))
        plt.close()


if __name__ == "__main__":
    main()
