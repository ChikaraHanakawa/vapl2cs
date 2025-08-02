import numpy as np
import glob
import os
import tqdm
import json

def read_vad_info(filename):
    vad_info = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'VAD':
                continue
            if line == 'TURN':
                break
            if 'BEGIN' in line:
                vad_info_local = []        
                continue
            if 'END' in line:
                vad_info.append(vad_info_local)
                continue
            start, end = [int(x)/1000 for x in line.split(',')]
            vad_info_local.append([start, end])
    return vad_info 

WASEDA_SOMA_DIR = "/autofs/diamond2/share/corpus/waseda_soma"
WASEDA_SOMA_WAV_DIR = os.path.join(WASEDA_SOMA_DIR, "wav_safia")
WASEDA_SOMA_DUR_DIR = os.path.join(WASEDA_SOMA_DIR, "duration_v2")
OUT_VAD_LIST_DIR = "./vad_list"

os.makedirs(OUT_VAD_LIST_DIR, exist_ok=True)
wav_paths = sorted(glob.glob(os.path.join(WASEDA_SOMA_WAV_DIR, "*.wav")))
for wav_path in tqdm.tqdm(wav_paths):
    basename = os.path.basename(wav_path)
    basename = os.path.splitext(basename)[0]
    dur_path = os.path.join(WASEDA_SOMA_DUR_DIR, basename + ".dur.txt")
    vad_list = read_vad_info(dur_path)
    out_path = os.path.join(OUT_VAD_LIST_DIR, basename + ".json")
    with open(out_path, 'w') as f:
        json.dump(vad_list, f)
