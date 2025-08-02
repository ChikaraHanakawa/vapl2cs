import glob
import pandas as pd
import os

def run(input_audio_vad, output_dir):
    df = pd.read_csv(input_audio_vad)
    # df の wav_file_path 列の basename の最初の3文字をとりだし，organization列に追加
    df["organization"] = df["audio_path"].map(lambda x: x.split("/")[-1][:3])
    # organization列に共通の値を持つもののうち，
    # 最後の1つだけを抽出して新たな df_test とする．
    # また，df_testに含まれるindexを除いたものをdf_trainとする．
    df_test = df.drop_duplicates("organization", keep="last")
    df_train = df.drop(df_test.index)
    # df_trainに含まれ，organization列に共通の値を持つ行が最も多い
    # 上位3つのorganizationのリストを取得
    organization_list = df_train["organization"].value_counts().index[:3].tolist()
    # df_trainに含まれ，organization列にorganization_listに含まれる値を持つもののうち，
    # 最後の行の要素を取り出しdf_validとする
    df_valid = df_train[df_train["organization"].isin(organization_list)].drop_duplicates("organization", keep="last")
    # df_validに含まれる行を
    # df_trainから取り除く
    df_train = df_train.drop(df_valid.index)
    df_train = df_train.drop("organization", axis=1)
    df_valid = df_valid.drop("organization", axis=1)
    df_test = df_test.drop("organization", axis=1)

    train_dir = os.path.join(output_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    df_train.to_csv(os.path.join(train_dir, "train_audio_vad.csv"), index=False)

    validation_dir = os.path.join(output_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    df_valid.to_csv(os.path.join(validation_dir, "val_audio_vad.csv"), index=False)

    test_dir = os.path.join(output_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    df_test.to_csv(os.path.join(test_dir, "test_audio_vad.csv"), index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_vad_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    run(args.audio_vad_csv, args.output_dir)
