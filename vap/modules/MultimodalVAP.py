import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

from vap.objective import VAPObjective
from vap.utils.utils import (
    everything_deterministic,
    vad_fill_silences,
    vad_omit_spikes,
)
from vap.modules.modules import ProjectionLayer
from vap.events.events import TurnTakingEvents, EventConfig

everything_deterministic()

OUT = dict[str, Tensor]


class MultimodalVAP(nn.Module):
    def __init__(
        self,
        audio_encoder: nn.Module,
        visual_encoder: nn.Module,
        transformer: nn.Module,
        bin_times: list[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        video_fps: int = 30,
        use_visual: bool = True,
    ):
        """
        Multimodal VAP model that integrates audio and visual features
        
        Args:
            audio_encoder: Audio encoder model
            visual_encoder: Visual encoder model for processing face images
            transformer: Transformer model for multimodal integration
            bin_times: Time bins for prediction
            frame_hz: Frame rate for audio features
            video_fps: Frame rate for video features
            use_visual: Whether to use visual features or audio only
        """
        super().__init__()
        self.audio_encoder = audio_encoder
        self.visual_encoder = visual_encoder
        self.transformer = transformer
        self.objective = VAPObjective(bin_times=bin_times, frame_hz=frame_hz)
        self.frame_hz = frame_hz
        self.video_fps = video_fps
        self.use_visual = use_visual
        
        # The dimensionality of the transformer model
        self.dim: int = getattr(self.transformer, "dim", 256)
        
        # Project audio features to transformer dimension if needed
        self.audio_projection = nn.Identity()
        if self.audio_encoder.dim != self.transformer.dim:
            self.audio_projection = ProjectionLayer(
                self.audio_encoder.dim, self.transformer.dim
            )
        
        # Project visual features to transformer dimension if needed
        self.visual_projection = nn.Identity()
        if self.use_visual and self.visual_encoder.dim != self.transformer.dim:
            self.visual_projection = ProjectionLayer(
                self.visual_encoder.dim, self.transformer.dim
            )
        
        # Outputs
        # Voice activity objective -> x1, x2 -> logits -> BCE
        self.va_classifier = nn.Linear(self.dim, 1)
        self.vap_head = nn.Linear(self.dim, self.objective.n_classes)

    @property
    def horizon_time(self) -> float:
        return self.objective.horizon_time

    @property
    def sample_rate(self) -> int:
        return self.audio_encoder.sample_rate

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def extract_labels(self, vad: Tensor) -> Tensor:
        return self.objective.get_labels(vad)

    def vad_loss(self, vad_output, vad):
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    def encode_audio(self, audio: torch.Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode audio inputs
        
        Args:
            audio: Audio tensor [B, 2, T]
            
        Returns:
            tuple of encoded features for speaker 1 and speaker 2
        """
        assert (
            audio.shape[1] == 2
        ), f"audio VAP ENCODER: {audio.shape} != (B, 2, n_samples)"
        x1 = self.audio_encoder(audio[:, :1])  # speaker 1
        x2 = self.audio_encoder(audio[:, 1:])  # speaker 2
        return x1, x2
    
    def encode_visual(self, images1: torch.Tensor, images2: torch.Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode visual inputs
        
        Args:
            images1: Images tensor for speaker 1 [B, F, C, H, W]
            images2: Images tensor for speaker 2 [B, F, C, H, W]
            
        Returns:
            tuple of encoded features for speaker 1 and speaker 2
        """
        batch_size, frames, channels, height, width = images1.shape
        
        # Reshape to process all frames at once
        images1_flat = images1.view(-1, channels, height, width)
        images2_flat = images2.view(-1, channels, height, width)
        
        # Encode all frames
        v1_flat = self.visual_encoder(images1_flat)  # [B*F, dim]
        v2_flat = self.visual_encoder(images2_flat)  # [B*F, dim]
        
        # Reshape back to batch and frames
        v1 = v1_flat.view(batch_size, frames, -1)  # [B, F, dim]
        v2 = v2_flat.view(batch_size, frames, -1)  # [B, F, dim]
        
        # Resample visual features to match audio frame rate if needed
        if self.frame_hz != self.video_fps:
            v1 = self._resample_visual_features(v1)
            v2 = self._resample_visual_features(v2)
            
        # Transpose to get shape [B, dim, F] to match audio encoder output format
        v1 = v1.transpose(1, 2)
        v2 = v2.transpose(1, 2)
        
        return v1, v2
    
    def _resample_visual_features(self, features: Tensor) -> Tensor:
        """
        Resample visual features to match audio frame rate
        
        Args:
            features: Visual features [B, F, dim]
            
        Returns:
            Resampled features [B, F', dim]
        """
        batch_size, frames, dim = features.shape
        
        # Calculate target length based on the ratio of frame rates
        target_length = int(frames * (self.frame_hz / self.video_fps))
        
        # Use interpolate to resample features
        features_transposed = features.transpose(1, 2)  # [B, dim, F]
        resampled = F.interpolate(
            features_transposed, 
            size=target_length, 
            mode='linear', 
            align_corners=False
        )
        
        return resampled.transpose(1, 2)  # [B, F', dim]

    def head(self, x: Tensor, x1: Tensor, x2: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply classification heads to the transformer outputs
        
        Args:
            x: Integrated features
            x1: Speaker 1 features 
            x2: Speaker 2 features
            
        Returns:
            tuple of VAP logits and VAD outputs
        """
        v1 = self.va_classifier(x1)
        v2 = self.va_classifier(x2)
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(x)
        return logits, vad

    def forward(self, waveform: Tensor, images1: Optional[Tensor] = None, 
                images2: Optional[Tensor] = None, attention: bool = False) -> OUT:
        """
        Forward pass for multimodal VAP
        
        Args:
            waveform: Audio waveform [B, 2, T]
            images1: Images for speaker 1 [B, F, C, H, W] or None if audio-only
            images2: Images for speaker 2 [B, F, C, H, W] or None if audio-only
            attention: Whether to return attention weights
            
        Returns:
            Dictionary with model outputs
        """
        # Encode audio
        a1, a2 = self.encode_audio(waveform)
        a1 = self.audio_projection(a1)
        a2 = self.audio_projection(a2)
        
        # Encode visual if available and enabled
        if self.use_visual and images1 is not None and images2 is not None:
            v1, v2 = self.encode_visual(images1, images2)
            v1 = self.visual_projection(v1)
            v2 = self.visual_projection(v2)
        else:
            v1, v2 = None, None
        
        # Pass through multimodal transformer
        out = self.transformer(a1, a2, v1, v2, attention=attention)
        
        # Apply classification heads
        logits, vad = self.head(out["x"], out["x1"], out["x2"])
        
        # Update output dictionary
        out["logits"] = logits
        out["vad"] = vad
        
        return out

    def entropy(self, probs: Tensor) -> Tensor:
        """
        Calculate entropy over each projection-window prediction (i.e. over
        frames/time) If we have C=256 possible states the maximum bit entropy
        is 8 (2^8 = 256) this means that the model have a one in 256 chance
        to randomly be right. The model can't do better than to uniformly
        guess each state, it has learned (less than) nothing. We want the
        model to have low entropy over the course of a dialog, "thinks it
        understands how the dialog is going", it's a measure of how close the
        information in the unseen data is to the knowledge encoded in the
        training data.
        """
        h = -probs * probs.log2()  # Entropy
        return h.sum(dim=-1).cpu()  # average entropy per frame

    def aggregate_probs(
        self,
        probs: Tensor,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ) -> dict[str, Tensor]:
        """
        Aggregate probabilities across different time bins
        
        Args:
            probs: Probabilities tensor
            now_lims: Bin indices for "now" time range
            future_lims: Bin indices for "future" time range
            
        Returns:
            Dictionary with aggregated probabilities
        """
        # first two bins
        p_now = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=now_lims[0], to_bin=now_lims[-1]
        ).cpu()
        p_future = self.objective.probs_next_speaker_aggregate(
            probs, from_bin=future_lims[0], to_bin=future_lims[1]
        ).cpu()
        # P over all
        max_idx = self.objective.n_bins - 1
        pa = self.objective.probs_next_speaker_aggregate(probs, 0, max_idx).cpu()
        p = []
        for i in range(0, max_idx + 1):
            p.append(self.objective.probs_next_speaker_aggregate(probs, i, i).cpu())
        p = torch.stack(p)
        return {
            "p_now": p_now,
            "p_future": p_future,
            "p_all": pa,
            "p": p,
        }

    @torch.inference_mode()
    def get_shift_probability(
        self, out: OUT, start_time: float, end_time: float, speaker
    ) -> dict[str, list[float]]:
        """
        Get shift probabilities (for classification) over the region `[start_time, end_time]`

        The `speaker` is the speaker before the silence, i.e. the speaker of the target IPU

        Shapes:
        out['p']:           (4, n_batch, n_frames)
        out['p_now']:       (n_batch, n_frames)
        out['p_future']:    (n_batch, n_frames)
        """
        region_start = int(start_time * self.frame_hz)
        region_end = int(end_time * self.frame_hz)
        ps = out["p"][..., region_start:region_end].mean(-1).cpu()
        pn = out["p_now"][..., region_start:region_end].mean(-1).cpu()
        pf = out["p_future"][..., region_start:region_end].mean(-1).cpu()

        batch_size = pn.shape[0]

        # if batch size == 1
        if batch_size == 1:
            speaker = [speaker]

        # Make all values 'shift probabilities'
        # The speaker is the speaker of the target IPU
        # A shift is the probability of the other speaker
        # The predictions values are always for the first speaker
        # So if the current speaker is speaker 1 then the probability of the default
        # speaker is the same as the shift-probability
        # However, if the current speaker is speaker 0 then the default probabilities
        # are HOLD probabilities, so we need to invert them
        for ii, spk in enumerate(speaker):
            if spk == 0:  # current speaker is 0, so shift means speaker 1
                # default probabilities are correct (probability of speaker 1)
                pass
            elif spk == 1:  # current speaker is 1, so shift means speaker 0
                # need to invert probabilities (1 - probability of speaker 1)
                pn[ii] = 1 - pn[ii]
                pf[ii] = 1 - pf[ii]
                ps[:, ii] = 1 - ps[:, ii]
            else:
                raise ValueError(f"Speaker must be 0 or 1, got {spk}")

        preds = {f"p{k+1}": v.tolist() for k, v in enumerate(ps)}
        preds["p_now"] = pn.tolist()
        preds["p_fut"] = pf.tolist()
        return preds

    @torch.inference_mode()
    def probs(
        self,
        waveform: Tensor,
        images1: Optional[Tensor] = None,
        images2: Optional[Tensor] = None,
        vad: Optional[Tensor] = None,
        now_lims: list[int] = [0, 1],
        future_lims: list[int] = [2, 3],
    ) -> OUT:
        """
        Get model predictions and probabilities
        
        Args:
            waveform: Audio waveform
            images1: Images for speaker 1 or None if audio-only
            images2: Images for speaker 2 or None if audio-only
            vad: Ground truth VAD if available
            now_lims: Bin indices for "now" time range
            future_lims: Bin indices for "future" time range
            
        Returns:
            Dictionary with model outputs and probabilities
        """
        out = self(waveform, images1, images2)
        probs = out["logits"].softmax(dim=-1)
        vap_vad = out["vad"].sigmoid()
        h = self.entropy(probs)
        ret = {
            "probs": probs,
            "vad": vap_vad,
            "logits": out["logits"],
            "H": h,
        }

        # Next speaker aggregate probs
        probs_agg = self.aggregate_probs(probs, now_lims, future_lims)
        ret.update(probs_agg)

        # If ground truth voice activity is known we can calculate the loss
        if vad is not None:
            ret["vad_loss"] = self.vad_loss(out["vad"], vad)
            ret["vap_loss"] = self.objective.loss_vap(out["logits"], self.extract_labels(vad))
            
        return ret

    @torch.inference_mode()
    def vad(
        self,
        waveform: Tensor,
        images1: Optional[Tensor] = None,
        images2: Optional[Tensor] = None,
        max_fill_silence_time: float = 0.02,
        max_omit_spike_time: float = 0.02,
        vad_cutoff: float = 0.5,
    ) -> Tensor:
        """
        Extract (binary) Voice Activity Detection from model
        
        Args:
            waveform: Audio waveform
            images1: Images for speaker 1 or None if audio-only
            images2: Images for speaker 2 or None if audio-only
            max_fill_silence_time: Maximum silence time to fill
            max_omit_spike_time: Maximum spike time to omit
            vad_cutoff: VAD threshold
            
        Returns:
            Binary VAD tensor
        """
        vad = (self(waveform, images1, images2)["vad"].sigmoid() >= vad_cutoff).float()
        for b in range(vad.shape[0]):
            for s in range(vad.shape[1]):
                # 1. Merge nearby speech regions (remove small silences)
                if max_fill_silence_time > 0:
                    vad[b, s] = vad_fill_silences(
                        vad[b, s], 
                        max_frames=int(max_fill_silence_time * self.frame_hz)
                    )
                # 2. Remove small speech regions (spikes)
                if max_omit_spike_time > 0:
                    vad[b, s] = vad_omit_spikes(
                        vad[b, s], 
                        max_frames=int(max_omit_spike_time * self.frame_hz)
                    )
        return vad
