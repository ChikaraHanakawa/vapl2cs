import os
from mcnr import do_multi_channel_noise_reduction
import soundfile as sf
import pandas as pd

def run(wav_check_csv_path, output_dir):
    df = pd.read_csv(wav_check_csv_path)

    for i, row in df.iterrows():
        input_wav_path = row["ファイル名（フルパス）"]
        exclude_check = row["除外"]
        print(exclude_check)
        if not pd.isna(exclude_check):
            continue
        output_wav_path = os.path.join(output_dir, os.path.basename(input_wav_path))
        print(f"{input_wav_path} -> {output_wav_path}")
        x, fs = sf.read(input_wav_path)
        x = x.T
        y = do_multi_channel_noise_reduction(x, chunk_size=16000)
        if not os.path.exists(os.path.dirname(output_wav_path)):
            os.makedirs(os.path.dirname(output_wav_path))
        sf.write(output_wav_path, y.T, fs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("wav_check_csv_path", type=str)
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()

    run(args.wav_check_csv_path, args.output_dir)