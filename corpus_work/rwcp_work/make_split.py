

def run(input_audio_vad, output_dir):
    import pandas as pd
    import os

    df_audio_vad = pd.read_csv(input_audio_vad)

    # df_audio_vad から audio_path 列のファイルパスに含まれる
    # ファイル名の先頭4文字を speaker 列として加える
    df_audio_vad["speaker"] = df_audio_vad["audio_path"].str.split("/").str[-1].str[:4]

    # それぞれのspeaker行の最終行を含むデータフレームと，
    # それ以外の行を含むデータフレームに分割
    df_audio_vad_test = df_audio_vad.groupby("speaker").tail(1)
    df_audio_vad_others = df_audio_vad.drop(df_audio_vad_test.index)

    # df_vad_others のうち，"C1_F_11.wav", "T1_M_11.wav" を含む行のデータフレームと
    # それ以外の行を含むデータフレームに分割
    df_audio_vad_C1_F_11 = df_audio_vad_others[df_audio_vad_others["audio_path"].str.contains("C1_F_11.wav")]
    df_audio_vad_T1_M_11 = df_audio_vad_others[df_audio_vad_others["audio_path"].str.contains("T1_M_11.wav")]
    df_audio_vad_validation = pd.concat([df_audio_vad_C1_F_11, df_audio_vad_T1_M_11])
    df_audio_vad_train = df_audio_vad_others.drop(df_audio_vad_validation.index)
    
    # 各データフレームからspeaker列を削除
    df_audio_vad_train = df_audio_vad_train.drop("speaker", axis=1)
    df_audio_vad_validation = df_audio_vad_validation.drop("speaker", axis=1)
    df_audio_vad_test = df_audio_vad_test.drop("speaker", axis=1)

    train_dir = os.path.join(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    df_audio_vad_train.to_csv(os.path.join(train_dir, "train_audio_vad.csv"), index=False)

    validation_dir = os.path.join(output_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    df_audio_vad_validation.to_csv(os.path.join(validation_dir, "val_audio_vad.csv"), index=False)

    test_dir = os.path.join(output_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    df_audio_vad_test.to_csv(os.path.join(test_dir, "test_audio_vad.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_vad_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    run(args.audio_vad_csv, args.output_dir)
