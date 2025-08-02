import torch
from torch.utils.data import Dataset
from pathlib import Path
from os.path import dirname
import pandas as pd
import random
from typing import Iterable, Optional

import tqdm

from torch import Tensor
from vap.data.datamodule import force_correct_nsamples
from vap.utils.audio import load_waveform, time_to_frames
from vap.utils.utils import vad_list_to_onehot, read_json, invalid_vad_list, get_dialog_states
from vap.events.events import Backchannel

VAD_LIST = list[list[list[float]]]


"""
Example of the file of the `--audio_vad_csv`

```csv
audio_path,vad_path
/audio/007/fe_03_00785.wav,/vad_lists/fe_03_00785.json
/audio/007/fe_03_00705.wav,/vad_lists/fe_03_00705.json
```
"""


def get_vad_list_min_max(vad_list: VAD_LIST, limit="max"):
    ch0 = vad_list[0]
    ch1 = vad_list[1]
    if limit == "max":
        ch0_max = -1
        if len(ch0) > 0:
            ch0_max = ch0[-1][-1]

        ch1_max = -1
        if len(ch1) > 0:
            ch1_max = ch1[-1][-1]

        m = max(ch0_max, ch1_max)

    else:
        ch0_min = 99999999
        if len(ch0) > 0:
            ch0_min = ch0[0][0]

        ch1_min = 99999999
        if len(ch1) > 0:
            ch1_min = ch1[0][0]

        m = min(ch0_min, ch1_min)
    return m


def extract_backchannels(vad_list: VAD_LIST, eventer: Backchannel) -> pd.DataFrame:
    def frame_to_time(frame: int, frame_hz: int = 50) -> float:
        return round(frame / frame_hz, 2)

    duration = max(vad_list[0][-1][-1], vad_list[1][-1][-1])
    vad = vad_list_to_onehot(vad_list, duration=duration, frame_hz=50)
    ee = eventer(vad.unsqueeze(0))  # add dummy batch dim
    data = []
    for name in ["pred_backchannel", "pred_backchannel_neg"]:
        all_events = ee[name][0]  # no batch dim
        for single_event in all_events:
            start, end, speaker = single_event
            label = "pred_backchannel" if name == "pred_backchannel" else "pred_backchannel_neg"
            data.append(
                {
                    "ipu_end": frame_to_time(start),
                    "tfo": frame_to_time(end - start),
                    "speaker": speaker,
                    "label": label,
                }
            )
    
    return pd.DataFrame(data)


def extract_ipu_classification(
    vad_list: VAD_LIST, 
    fill_time: float = 0.1,
    detect_backchannels: bool = False,
    max_bc_duration: float = 2.0
) -> pd.DataFrame:
    def get_tfo_other(
        ipu_end: float, ipus_other: list[tuple[float, float]]
    ) -> tuple[bool, float, float, float]:
        # Find overlapping region
        for ss, ee in ipus_other:
            if ss < ipu_end and ipu_end < ee:
                tfo = ss - ipu_end  # a negative TFO value
                return True, tfo, ss, ee
            if ipu_end < ss:
                tfo = ss - ipu_end
                # start is after end
                return False, tfo, ss, ee
        return False, 9999.0, 0.0, 0.0
    
    def is_backchannel(duration, next_duration):
        # A backchannel is typically short and doesn't lead to a turn switch
        # Simple heuristic: short utterances that don't result in turn-taking
        return duration <= max_bc_duration and next_duration > 0

    def vad_list_fill_silences(vad_list: VAD_LIST, fill_time: float = 0.02) -> VAD_LIST:
        """
        A channel is a list of [start, end] times.
        This functions joins segments separated by less than `fill_time` seconds.
        """
        new_vad_list = []
        for ch_vad in vad_list:
            filled_intervals = []
            if not ch_vad:  # Handle empty channel
                new_vad_list.append([])
                continue
                
            current_interval = ch_vad[0]
            for next_interval in ch_vad[1:]:
                silence_between = next_interval[0] - current_interval[1]
                if silence_between <= fill_time:
                    # Extend current_interval to include next_interval
                    current_interval[1] = next_interval[1]
                else:
                    # Append current_interval to filled_intervals and start a new one
                    filled_intervals.append(current_interval)
                    current_interval = next_interval
            # Don't forget to add the last interval
            filled_intervals.append(current_interval)
            new_vad_list.append(filled_intervals)
        return new_vad_list

    ipu_list = vad_list_fill_silences(vad_list, fill_time=fill_time)
    ipu_ends = []
    
    for channel in range(2):
        other_channel = 0 if channel == 1 else 1
        ipus = ipu_list[channel]
        ipus_other = ipu_list[other_channel]
        
        for i, (s, end) in enumerate(ipus[:-1]):
            next_start = ipus[i + 1][0]
            next_end = ipus[i + 1][1]
            tfo_same = next_start - end
            
            # Get information about other speaker's IPUs
            is_overlap, tfo_other, other_start, other_end = get_tfo_other(end, ipus_other)
            
            # Calculate durations for backchannel detection
            current_duration = end - s
            next_duration = next_end - next_start
            other_duration = other_end - other_start if other_end > 0 else 0
            
            if is_overlap:
                ipu_ends.append(
                    {
                        "ipu_end": end,
                        "tfo": tfo_other,
                        "speaker": channel,
                        "label": "overlap",
                    }
                )
            elif detect_backchannels and is_backchannel(current_duration, next_duration):
                # This is potentially a backchannel
                ipu_ends.append(
                    {
                        "ipu_end": end,
                        "tfo": tfo_same if tfo_same < tfo_other else tfo_other,
                        "speaker": channel,
                        "label": "pred_backchannel",
                        "duration": current_duration,
                    }
                )
    
    ipu_ends.sort(key=lambda x: x["ipu_end"])
    return pd.DataFrame(ipu_ends)


def create_classification_dset(
    audio_vad_path: str,
    output: str,
    pre_cond_time: float = 1.0,  # single speaker prior to silence
    post_cond_time: float = 2.0,  # single speaker post silence
    min_silence_time: float = 0.1,  # minimum reaction time / silence duration
    ipu_based_events: bool = False,
    # Backchannel specific parameters
    prediction_region_time: float = 1.0,
    min_context_time: float = 1.0,
    negative_pad_left_time: float = 1.0,
    negative_pad_right_time: float = 1.0,
    max_bc_duration: float = 2.0,
):

    eventer = Backchannel(
        pre_cond_time=pre_cond_time,
        post_cond_time=post_cond_time,
        prediction_region_time=prediction_region_time,
        min_context_time=min_context_time,
        negative_pad_left_time=negative_pad_left_time,
        negative_pad_right_time=negative_pad_right_time,
        max_bc_duration=max_bc_duration,
        max_time=999999,
        frame_hz=50,
    )

    # read csv
    audio_vad = pd.read_csv(audio_vad_path)

    all_dfs = []
    skipped = []
    for _, row in tqdm.tqdm(
        audio_vad.iterrows(), total=len(audio_vad), desc="Extracting event dataset"
    ):
        vad_list = read_json(row.vad_path)

        if invalid_vad_list(vad_list):
            skipped.append(row.vad_path)
            continue

        if ipu_based_events:
            c = extract_ipu_classification(vad_list, fill_time=min_silence_time)
        else:
            c = extract_backchannels(vad_list, eventer)
        c["audio_path"] = row.audio_path
        c["vad_path"] = row.vad_path
        all_dfs.append(c)
    c = pd.concat(all_dfs, ignore_index=True)

    if len(skipped) > 0:
        print("Skipped: ", len(skipped))
        with open("/tmp/dset_event_skipped_vad.txt", "w") as f:
            f.write("\n".join(skipped))
        print("See -> /tmp/dset_event_skipped_vad.txt")
        print()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    c.to_csv(output, index=False)
    ev_type = "IPU" if ipu_based_events else "pred_backchannel_neg"
    print(f"Saved {len(c)} {ev_type} -> ", output)


class VAPClassificationDataset(Dataset):
    def __init__(
        self,
        df_path: str,
        context: float = 10.0,
        post_silence: float = 1.0,
        min_event_silence: float = 0,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
    ) -> None:
        self.df_path = df_path
        self.df = pd.read_csv(df_path)
        
        # Filter overlaps if not dealing with backchannel data
        if not any(self.df["label"].isin(["pred_backchannel", "pred_backchannel_neg"])):
            self.df = self.df[self.df["label"] != "overlap"]

        if min_event_silence > 0:
            self.df = self.df[self.df["tfo"] >= min_event_silence]

        self.artificial_silence = post_silence
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.mono = mono
        self.context = context
        self.n_samples = int(self.context * self.sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        d = self.df.iloc[idx]

        start_time = 0
        if d["ipu_end"] > self.context:
            start_time = d["ipu_end"] - self.context
        w, _ = load_waveform(
            d["audio_path"],
            start_time=start_time,
            end_time=d["ipu_end"],
            sample_rate=self.sample_rate,
            mono=self.mono,
        )
        n_channels = w.shape[0]
        if start_time == 0:
            diff = self.context - d["ipu_end"]
            # Add silence to the beginning
            silence_prefix = torch.zeros(n_channels, int(diff * self.sample_rate))
            w = torch.cat((silence_prefix, w), dim=-1)
        # Ensure correct duration
        # Some clips (20s) becomes
        # [2, 320002] insted of [2, 320000]
        # breaking the batching
        n_samples = int(self.context * self.sample_rate)
        w = force_correct_nsamples(w, n_samples)

        # Add artificial silence
        silence_suffix = torch.zeros(
            (n_channels, int(self.artificial_silence * self.sample_rate))
        )
        w = torch.cat((w, silence_suffix), dim=-1)
        return {
            "session": d.get("session", ""),
            "waveform": w,
            "label": d["label"],
            "tfo": d["tfo"],
            "speaker": d["speaker"],
            "dataset": d.get("dataset", ""),
        }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--audio_vad_csv", type=str)
    parser.add_argument(
        "--output", type=str, default="data/classification/audio_vad_hs.csv"
    )
    parser.add_argument("--pre_cond_time", type=float, default=1.0)
    parser.add_argument("--post_cond_time", type=float, default=2.0)
    parser.add_argument("--min_silence_time", type=float, default=0.1)
    parser.add_argument("--ipu_based_events", action="store_true")
    # Backchannel specific parameters
    parser.add_argument("--prediction_region_time", type=float, default=1.0)
    parser.add_argument("--min_context_time", type=float, default=1.0)
    parser.add_argument("--negative_pad_left_time", type=float, default=1.0)
    parser.add_argument("--negative_pad_right_time", type=float, default=1.0)
    parser.add_argument("--max_bc_duration", type=float, default=2.0)
    
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    create_classification_dset(
        audio_vad_path=args.audio_vad_csv,
        output=args.output,
        pre_cond_time=args.pre_cond_time,
        post_cond_time=args.post_cond_time,
        min_silence_time=args.min_silence_time,
        ipu_based_events=args.ipu_based_events,
        prediction_region_time=args.prediction_region_time,
        min_context_time=args.min_context_time,
        negative_pad_left_time=args.negative_pad_left_time,
        negative_pad_right_time=args.negative_pad_right_time,
        max_bc_duration=args.max_bc_duration,
    )