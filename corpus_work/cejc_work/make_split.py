import pandas as pd
import os

def run(split_csv_path, audio_vad_path, output_dir):
    split_df = pd.read_csv(split_csv_path)
    split_df = split_df[['会話ID', 'split']]

    audio_vad_df = pd.read_csv(audio_vad_path)

    # audio_vad_df の audio_path 列にあるフルパスから
    # ファイル名の本体（拡張子なし）を取得し，kaiwa_id 列に追加
    audio_vad_df["kaiwa_id"] = audio_vad_df["audio_path"].map(lambda x: os.path.basename(x).split(".")[0])

    # audio_vad_df の行のうち，
    # kaiwa_id で split_df を検索した時に，split に test がある行を抽出
    test_df = audio_vad_df[audio_vad_df["kaiwa_id"].isin(split_df[split_df["split"] == "test"]["会話ID"])]
    test_df = test_df.drop("kaiwa_id", axis=1)

    # audio_vad_df の行のうち，
    # kaiwa_id で split_df を検索した時に，split に valid がある行を抽出
    val_df = audio_vad_df[audio_vad_df["kaiwa_id"].isin(split_df[split_df["split"] == "valid"]["会話ID"])]
    val_df = val_df.drop("kaiwa_id", axis=1)


    # audio_vad_df の行のうち，
    # test_dfやval_dfに含まれない行を抽出
    # indexで絞り込めるはず
    train_df = audio_vad_df.drop(test_df.index)
    train_df = train_df.drop(val_df.index)
    train_df = train_df.drop("kaiwa_id", axis=1)
    
    train_dir = os.path.join(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    train_df.to_csv(os.path.join(train_dir, "train_audio_vad.csv"), index=False)

    validation_dir = os.path.join(output_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    val_df.to_csv(os.path.join(validation_dir, "val_audio_vad.csv"), index=False)

    test_dir = os.path.join(output_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_df.to_csv(os.path.join(test_dir, "test_audio_vad.csv"), index=False)

    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_csv", type=str, required=True)
    parser.add_argument("--audio_vad_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    run(args.split_csv, args.audio_vad_csv, args.output_dir)    
    