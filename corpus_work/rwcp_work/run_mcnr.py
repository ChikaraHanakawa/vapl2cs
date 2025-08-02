import os
import glob
from mcnr import do_multi_channel_noise_reduction
import soundfile as sf

def run(input_top_dir, output_dir):
    # input_top_dir 以下にある "*.wav" にマッチするパスを，サブディレクトリも
    # 含めて全て取得する
    wav_files = sorted(glob.glob(os.path.join(input_top_dir, "**/*.wav"), recursive=True))


    # wav_files の basename のリストを作成し，
    # "*H.wav"にマッチするものを抽出する
    wav_files_H = [os.path.basename(f) for f in wav_files if f.endswith("H.wav")]
    # wav_files_H の basename から，"H" を取り除いたものを，作成する
    # 例: "aaaH.wav" -> "aaa.wav"
    wav_files_H = [f.replace("H.wav", ".wav") for f in wav_files_H]

    for input_wav_path in wav_files:
        basename = os.path.basename(input_wav_path)
        if basename in wav_files_H:
            # "H" が含まれる wav ファイルの場合は，処理をスキップ
            continue
        # basename から H を取り除く (例: "aaaH.wav" -> "aaa.wav")
        basename = basename.replace("H.wav", ".wav")
        output_wav_path = os.path.join(output_dir, basename)
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
    # 入力ディレクトリ，複数の入力を受け付ける
    parser.add_argument("--input_dir", "-i", type=str, action="append")
    # 出力ディレクトリ
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()

    for input_dir in args.input_dir:
        run(input_dir, args.output_dir)