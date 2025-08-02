#!/usr/bin/env python

import torch
import logging
import hydra
import os
import sys
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything
from lightning.pytorch import Trainer

log: logging.Logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision(precision="medium")

@hydra.main(version_base=None, config_path="conf", config_name="multimodal_vap_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training and evaluating the Multimodal VAP model
    
    Args:
        cfg: Configuration object from Hydra
    """
    # Set random seed
    seed = cfg.get("seed", 0)
    seed_everything(seed, workers=True)
    
    # Log configuration
    log.info(OmegaConf.to_yaml(cfg))
    
    # Instantiate model and datamodule
    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    
    # Load from checkpoint if specified
    if getattr(cfg, "pretrained_checkpoint_path", None):
        module = module.load_from_checkpoint(
            checkpoint_path=cfg.pretrained_checkpoint_path
        )
        print(f"Loaded from checkpoint: {cfg.pretrained_checkpoint_path}")
        
        # Add metrics if needed
        if getattr(cfg.module, "val_metric", False):
            val_metric = instantiate(cfg.module.val_metric)
            module.val_metric = val_metric
            print("Added val metrics")
            
        if getattr(cfg.module, "test_metric", False):
            test_metric = instantiate(cfg.module.test_metric)
            module.test_metric = test_metric
            print("Added test metrics")
            
        # Prompt user to confirm
        if not cfg.get("no_prompt", False):
            input("Press enter to continue: ")
    
    # Create trainer
    if getattr(cfg, "debug", False):
        trainer = Trainer(fast_dev_run=True)
    else:
        trainer = instantiate(cfg.trainer)
    
    # Print system info
    print(f"CPUs: {os.cpu_count()}")
    print(f"Pytorch Threads: {torch.get_num_threads()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Devices: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.get_device_name()}")
    
    # Run training if not in eval-only mode
    if not cfg.get("eval_only", False):
        trainer.fit(module, datamodule=datamodule)
    
        # Run testing if checkpoint exists
    if hasattr(module, "test_metric") and (cfg.get("eval_only", False) or trainer.checkpoint_callback.best_model_path):
        checkpoint_path = cfg.get("pretrained_checkpoint_path") if cfg.get("eval_only", False) else trainer.checkpoint_callback.best_model_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading best checkpoint for testing: {checkpoint_path}")
            # Load best checkpoint for testing
            module = module.__class__.load_from_checkpoint(checkpoint_path, model=module.model, strict=False)
            
            # Make sure we have test metrics
            if hasattr(cfg.module, "test_metric") and cfg.module.test_metric:
                print("Adding test metrics to module...")
                try:
                    module.test_metric = instantiate(cfg.module.test_metric)
                    print(f"Successfully instantiated test_metric: {type(module.test_metric).__name__}")
                except Exception as e:
                    print(f"Error initializing test_metric: {str(e)}")
                    module.test_metric = None
            else:
                # If test_metric not defined in config, use the same configuration as val_metric
                if hasattr(cfg.module, "val_metric") and cfg.module.val_metric:
                    print("No test_metric found in config. Using val_metric configuration for testing...")
                    try:
                        module.test_metric = instantiate(cfg.module.val_metric)
                        print(f"Successfully instantiated test_metric from val_metric: {type(module.test_metric).__name__}")
                    except Exception as e:
                        print(f"Error initializing test_metric from val_metric: {str(e)}")
                        module.test_metric = None
            # Run testing
            trainer.test(module, datamodule=datamodule)
            
            # Save outputs
            output_dir = cfg.get("output_dir", Path.cwd() / "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Collect test results - only if test_metric exists and is not None
            results = {}
            if hasattr(module, "test_metric") and module.test_metric is not None:
                print("Collecting test results from test_metric...")
                try:
                    # VAPMetricクラスはget_results()ではなくcompute()メソッドを使用します
                    metrics = module.test_metric.compute()
                    # 計算後にメトリックをリセットします
                    module.test_metric.reset()
                    
                    # 結果を整形して出力
                    print("\n===== Test Results =====")
                    for event_name, score in metrics.items():
                        print(f"\n--- {event_name} ---")
                        print(f"F1 Score: {score['f1']:.4f}")
                        print(f"Class 0 Accuracy: {score['acc'][0]:.4f}")
                        print(f"Class 1 Accuracy: {score['acc'][1]:.4f}")
                    
                    # 結果をJSONに変換
                    results = {}
                    for event_name, score in metrics.items():
                        results[f"test_f1_{event_name}"] = score["f1"].item()
                        results[f"test_acc_{event_name}_0"] = score["acc"][0].item()
                        results[f"test_acc_{event_name}_1"] = score["acc"][1].item()
                    
                    # イベント名を人間が読みやすい形式に変換するための説明を追加
                    results["event_name_explanations"] = {
                        "hs": "Hold vs Shift classification",
                        "ls": "Long vs Short utterance classification",
                        "sp": "Shift Prediction",
                        "bp": "Backchannel Prediction"
                    }
                except Exception as e:
                    print(f"Error collecting test results: {str(e)}")
                    results = {"error": str(e)}
            else:
                print("Warning: test_metric is not available or is None. No detailed results will be saved.")
            
            # Save results
            import json
            with open(os.path.join(output_dir, "test_results.json"), "w") as f:
                json.dump(results, f, indent=2)
                
            print(f"Test results saved to {output_dir}/test_results.json")


if __name__ == "__main__":
    main()
