import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import tqdm

from vap.data.datamodule import force_correct_nsamples
from vap.utils.audio import load_waveform
from vap.utils.utils import vad_list_to_onehot, read_json, invalid_vad_list
from vap.events.events import HoldShift

VAD_LIST = list[list[list[float]]]


def extract_prediction_regions(vad_list: VAD_LIST, eventer: HoldShift) -> pd.DataFrame:
    """
    Extract prediction regions (pred_shift and pred_hold) from VAD list
    """
    def frame_to_time(frame: int, frame_hz: int = 50) -> float:
        return round(frame / frame_hz, 2)

    duration = max(vad_list[0][-1][-1], vad_list[1][-1][-1]) if vad_list[0] and vad_list[1] else 0
    if duration == 0:
        return pd.DataFrame(columns=["region_start", "region_end", "speaker", "label"])
    
    vad = vad_list_to_onehot(vad_list, duration=duration, frame_hz=50)
    ee = eventer(vad.unsqueeze(0))  # add dummy batch dim
    
    data = []
    for name in ["pred_shift", "pred_hold"]:
        label = "pred_shift" if name == "pred_shift" else "pred_hold"
        all_regions = ee[name][0]  # no batch dim
        
        for single_region in all_regions:
            start, end, next_speaker = single_region
            data.append({
                "ipu_end": frame_to_time(start),
                "tfo": frame_to_time(end - start),
                "speaker": next_speaker,
                "label": label
            })
    
    return pd.DataFrame(data)


def create_prediction_dset(
    audio_vad_path: str,
    output: str,
    pre_cond_time: float = 1.0,  # single speaker prior to silence
    post_cond_time: float = 2.0,  # single speaker post silence
    min_silence_time: float = 0.1,  # minimum reaction time / silence duration
    prediction_region_time: float = 0.5,  # prediction region duration
    prediction_region_on_active: bool = True,  # whether prediction region is on active speaker
    long_onset_condition_time: float = 0.5,  # condition for long onset
    long_onset_region_time: float = 0.5,  # region for long onset
):
    eventer = HoldShift(
        pre_cond_time=pre_cond_time,
        post_cond_time=post_cond_time,
        min_silence_time=min_silence_time,
        prediction_region_time=prediction_region_time,
        prediction_region_on_active=prediction_region_on_active,
        long_onset_condition_time=long_onset_condition_time,
        long_onset_region_time=long_onset_region_time,
        min_context_time=0,
        max_time=999999,
        frame_hz=50,
    )

    # read csv
    audio_vad = pd.read_csv(audio_vad_path)

    all_dfs = []
    skipped = []
    for _, row in tqdm.tqdm(
        audio_vad.iterrows(), total=len(audio_vad), desc="Extracting prediction regions"
    ):
        vad_list = read_json(row.vad_path)

        if invalid_vad_list(vad_list):
            skipped.append(row.vad_path)
            continue

        # Extract prediction regions
        pred_regions = extract_prediction_regions(vad_list, eventer)
        if len(pred_regions) == 0:
            continue
            
        pred_regions["audio_path"] = row.audio_path
        pred_regions["vad_path"] = row.vad_path
        all_dfs.append(pred_regions)
    
    if not all_dfs:
        print("No prediction regions found in the dataset.")
        return
        
    result_df = pd.concat(all_dfs, ignore_index=True)

    if len(skipped) > 0:
        print("Skipped: ", len(skipped))
        with open("/tmp/dset_predhc_skipped_vad.txt", "w") as f:
            f.write("\n".join(skipped))
        print("See -> /tmp/dset_predhc_skipped_vad.txt")
        print()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output, index=False)
    print(f"Saved {len(result_df)} prediction regions -> ", output)


class VAPPredictionDataset(Dataset):
    def __init__(
        self,
        df_path: str,
        context: float = 5.0,  # Context window before prediction region
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
    ) -> None:
        self.df_path = df_path
        self.df = pd.read_csv(df_path)
        
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.mono = mono
        self.context = context
        self.n_samples = int(self.context * self.sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        d = self.df.iloc[idx]

        # Calculate region duration
        region_duration = d["region_end"] - d["region_start"]
        
        # Load audio starting from context before prediction region to the end of region
        start_time = max(0, d["region_start"] - self.context)
        end_time = d["region_end"]
        
        w, _ = load_waveform(
            d["audio_path"],
            start_time=start_time,
            end_time=end_time,
            sample_rate=self.sample_rate,
            mono=self.mono,
        )
        
        n_channels = w.shape[0]
        
        # If we couldn't get full context (because region_start < context)
        # pad with silence at the beginning
        if start_time == 0 and d["region_start"] < self.context:
            missing_context = self.context - d["region_start"]
            silence_prefix = torch.zeros(n_channels, int(missing_context * self.sample_rate))
            w = torch.cat((silence_prefix, w), dim=-1)
            
        # Ensure correct duration for the entire segment
        total_expected_samples = int((self.context + region_duration) * self.sample_rate)
        w = force_correct_nsamples(w, total_expected_samples)
        
        # Calculate the index where prediction region starts in the waveform
        pred_start_idx = int(self.context * self.sample_rate)
        
        return {
            "session": d.get("session", ""),
            "waveform": w,
            "label": d["label"],
            "speaker": d["speaker"],
            "region_start": d["region_start"],
            "region_end": d["region_end"],
            "pred_start_idx": pred_start_idx,
            "dataset": d.get("dataset", ""),
        }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--audio_vad_csv", type=str)
    parser.add_argument(
        "--output", type=str, default="data/prediction/audio_vad_predhc.csv"
    )
    parser.add_argument("--pre_cond_time", type=float, default=1.0)
    parser.add_argument("--post_cond_time", type=float, default=2.0)
    parser.add_argument("--min_silence_time", type=float, default=0.1)
    parser.add_argument("--prediction_region_time", type=float, default=0.5)
    parser.add_argument("--prediction_region_on_active", action="store_true")
    parser.add_argument("--long_onset_condition_time", type=float, default=0.5)
    parser.add_argument("--long_onset_region_time", type=float, default=0.5)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    create_prediction_dset(
        audio_vad_path=args.audio_vad_csv,
        output=args.output,
        pre_cond_time=args.pre_cond_time,
        post_cond_time=args.post_cond_time,
        min_silence_time=args.min_silence_time,
        prediction_region_time=args.prediction_region_time,
        prediction_region_on_active=args.prediction_region_on_active,
        long_onset_condition_time=args.long_onset_condition_time,
        long_onset_region_time=args.long_onset_region_time,
    )