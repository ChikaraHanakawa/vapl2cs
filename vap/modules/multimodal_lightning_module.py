import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

import lightning as L
from typing import Optional, Mapping, Iterable, Callable, Dict, List, Union, Tuple

from vap.metrics import VAPMetric
from vap.modules.MultimodalVAP import MultimodalVAP
from vap.utils.utils import everything_deterministic

from vap.events.events import TurnTakingEvents
from vap.objective import VAPObjective


Batch = Mapping[str, Tensor]

everything_deterministic()


class MultimodalVAPModule(L.LightningModule):
    def __init__(
        self,
        model: MultimodalVAP,
        optim_fn: Optional[Callable[[Iterable[Parameter]], Optimizer]] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        train_metric: Optional[VAPMetric] = None,
        val_metric: Optional[VAPMetric] = None,
        test_metric: Optional[VAPMetric] = None,
        use_images: bool = True,
        use_visual_masking: bool = True,
        visual_masking_prob: float = 0.3,
    ):
        """
        Lightning module for Multimodal VAP
        
        Args:
            model: The multimodal VAP model
            optim_fn: Optimizer function
            lr_scheduler: Learning rate scheduler
            train_metric: Training metric
            val_metric: Validation metric
            test_metric: Test metric
            use_images: Whether to use images in the model
            use_visual_masking: Whether to apply random masking to visual features during training
            visual_masking_prob: Probability of masking visual features
        """
        super().__init__()
        self.model = model
        self.optim: Optional[Optimizer] = (
            optim_fn(self.model.parameters()) if optim_fn else None
        )
        self.lr_scheduler: Optional[_LRScheduler] = (
            lr_scheduler(self.optim) if lr_scheduler else None
        )
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.use_images = use_images
        self.use_visual_masking = use_visual_masking
        self.visual_masking_prob = visual_masking_prob
        self.save_hyperparameters(ignore=["model"])

    def forward(self, waveform: Tensor, images1: Optional[Tensor] = None, images2: Optional[Tensor] = None) -> dict[str, Tensor]:
        """
        Forward pass through the model
        """
        return self.model(waveform, images1, images2)

    @staticmethod
    def load_model(path: str) -> MultimodalVAP:
        """
        Load model from checkpoint
        """
        return MultimodalVAPModule.load_from_checkpoint(path).model

    def configure_optimizers(self) -> dict:
        """
        Configure optimizers and learning rate schedulers
        """
        lr_scheduler = {
            "scheduler": self.lr_scheduler,
            "monitor": "val_loss",
        }
        return {"optimizer": self.optim, "lr_scheduler": lr_scheduler}

    def metric_update(self, logits, vad, split: str = "val"):
        """
        Update metrics with batch results
        """
        m = getattr(self, f"{split}_metric", None)
        if m:
            probs = self.model.objective.get_probs(logits)
            m.update_batch(probs, vad)

    def metric_finalize(self, split: str = "val") -> None:
        """
        Compute and log final metrics
        """
        m = getattr(self, f"{split}_metric", None)
        if m:
            scores = m.compute()
            m.reset()
            for event_name, score in scores.items():
                self.log(f"{split}_f1_{event_name}", score["f1"], sync_dist=True)
                self.log(f"{split}_acc_{event_name}_0", score["acc"][0], sync_dist=True)
                self.log(f"{split}_acc_{event_name}_1", score["acc"][1], sync_dist=True)

    def _apply_visual_masking(self, images1: Tensor, images2: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply random masking to visual features during training
        
        Args:
            images1: Visual features for speaker 1 [B, F, C, H, W]
            images2: Visual features for speaker 2 [B, F, C, H, W]
            
        Returns:
            Masked visual features
        """
        if not self.use_visual_masking or torch.rand(1).item() > self.visual_masking_prob:
            return images1, images2
            
        batch_size, frames, channels, height, width = images1.shape
        
        # Create a random binary mask for each sample in the batch
        mask1 = torch.rand(batch_size, frames, 1, 1, 1, device=images1.device) > 0.5
        mask2 = torch.rand(batch_size, frames, 1, 1, 1, device=images2.device) > 0.5
        
        # Apply the masks
        masked_images1 = images1 * mask1.to(images1.dtype)
        masked_images2 = images2 * mask2.to(images2.dtype)
        
        return masked_images1, masked_images2

    def _step(
        self, batch: Batch, split: str = "train", reduction: str = "mean"
    ) -> Mapping[str, torch.Tensor]:
        """
        Common step function for training, validation, and testing
        
        Args:
            batch: Input batch containing 'waveform', 'vad', and optionally 'images1', 'images2'
            split: Which split this is ('train', 'val', 'test')
            reduction: Reduction method for loss calculation
            
        Returns:
            Dictionary with model outputs and losses
        """
        # Extract inputs from batch
        waveform = batch["waveform"]
        images1 = batch.get("images1", None) if self.use_images else None
        images2 = batch.get("images2", None) if self.use_images else None
        has_images = batch.get("has_images", False) if self.use_images else False
        
        # Set images to None if they're not being used or not available
        if not self.use_images or not has_images:
            images1, images2 = None, None
        
        # Apply visual masking during training
        if split == "train" and images1 is not None and images2 is not None:
            images1, images2 = self._apply_visual_masking(images1, images2)
        
        # Extract labels for the objective
        vad = batch["vad"]
        labels = self.model.extract_labels(vad)
        
        # Forward pass through the model
        out = self(waveform, images1, images2)
        
        # Update metrics if available
        if split.startswith("test"):
            self.metric_update(out["logits"], vad, split=split)
        
        # Adjust label length if needed (for models like HuBERT that may return different frame lengths)
        if labels.shape[1] != out["logits"].shape[1]:
            labels = labels[:, : out["logits"].shape[1]]
        
        # Calculate losses
        out["vap_loss"] = self.model.objective.loss_vap(
            out["logits"], labels, reduction=reduction
        )
        out["va_loss"] = self.model.objective.loss_vad(out["vad"], batch["vad"])
        
        # Update metrics
        self.metric_update(out["logits"], batch["vad"], split=split)

        # Log results
        batch_size = batch["waveform"].shape[0]
        self.log(
            f"{split}_loss", out["vap_loss"], batch_size=batch_size, sync_dist=True
        )
        self.log(
            f"{split}_loss_va",
            out["va_loss"],
            batch_size=batch_size,
            sync_dist=True,
        )
        
        return out

    def training_step(self, batch: Batch, *args, **kwargs):
        """
        Training step
        """
        out = self._step(batch, split="train")
        loss = out["vap_loss"] + out["va_loss"]
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs):
        """
        Validation step
        """
        _ = self._step(batch, split="val")

    def test_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0, *args, **kwargs):
        """
        Test step with support for multiple test dataloaders
        """
        # Check if we're using single-speaker evaluation mode
        eval_mode = dataloader_idx > 0 if isinstance(dataloader_idx, int) else False
        
        if eval_mode:
            # Use the appropriate waveform for single-speaker evaluation
            waveform = batch.get("eval_waveform", batch["waveform"])
            suffix = f"_spk{dataloader_idx - 1}" if dataloader_idx > 0 else ""
        else:
            waveform = batch["waveform"]
            suffix = ""
        
        # Update the batch with the appropriate waveform
        test_batch = {**batch, "waveform": waveform}
        
        # Process the batch
        out = self._step(test_batch, split=f"test{suffix}")
        
        # Explicitly update the test metrics if they exist
        if hasattr(self, "test_metric") and self.test_metric is not None:
            self.test_metric.update_batch(
                self.model.objective.get_probs(out["logits"]), 
                batch["vad"]
            )
        
        # Return the model outputs for further analysis if needed
        return out

    def on_train_epoch_end(self) -> None:
        """
        End of training epoch
        """
        self.metric_finalize(split="train")

    def on_validation_epoch_end(self) -> None:
        """
        End of validation epoch
        """
        self.metric_finalize(split="val")

    def on_test_epoch_end(self) -> None:
        """
        End of test epoch
        """
        # Handle potentially multiple test metrics if using multiple test dataloaders
        self.metric_finalize(split="test")
        if hasattr(self, "test_metric_spk0"):
            self.metric_finalize(split="test_spk0")
        if hasattr(self, "test_metric_spk1"):
            self.metric_finalize(split="test_spk1")
