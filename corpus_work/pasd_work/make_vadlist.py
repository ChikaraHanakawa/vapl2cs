import glob
import argparse
import os
import json
import soundfile as sf
from run_vad import run_vad

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1) 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def run():
    parser = argparse.ArgumentParser()

    default_wav_dir = "/autofs/diamond2/share/users/fujie/work/vap_data/pasd/data/wav"
    default_vad_dir = "/autofs/diamond2/share/users/fujie/work/vap_data/pasd/data/vad"

    parser.add_argument("--wav_dir", "-i", type=str, default=default_wav_dir)
    parser.add_argument("--vad_dir", "-o", type=str, default=default_vad_dir)

    args = parser.parse_args()

    wav_files = sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    for wav_file in wav_files:
        print(wav_file)
        audio, sr = sf.read(wav_file)
        vad = run_vad(audio)

        vad_file = os.path.join(args.vad_dir, os.path.basename(wav_file).replace(".wav", ".json"))

        if not os.path.exists(os.path.dirname(vad_file)):
            os.makedirs(os.path.dirname(vad_file))
        
        with open(vad_file, "w") as f:
            json.dump(vad, f)

if __name__ == "__main__":
    run()