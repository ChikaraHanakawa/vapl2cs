import json
import os
import pandas as pd

VAD_LIST = "./vad_list"
CSV = "~/project/2024/test/StreamVAP/data/splits_tabidachi/sliding_window_dset/train.csv"

def validate_structure(data):
    """
    入力データが指定された入力規則を満たしているかを確認します。
    規則：
    1. データは3次元配列であること。
    2. 1次元目は必ず2次元の配列であること。
    3. 3次元目は常に2次元配列であること。

    :param data: 検証するデータ
    :return: bool 検証結果（True: 規則を満たしている、False: 満たしていない）
    """
    # 1次元目がリストであるか
    if not isinstance(data, list):
        return False
    
    # 2次元目を検証
    for dim2 in data:
        if not isinstance(dim2, list):
            return False
        # 3次元目を検証
        for dim3 in dim2:
            if not isinstance(dim3, list) or len(dim3) != 2:
                return False
    return True

def check_csv_content(csv_path):
    df = pd.read_csv(csv_path)
    print("CSV columns:", df.columns)
    print("\nSample VAD lists:")
    for idx, row in df.iterrows():
        if row.isnull().all():
            print(f"空の配列が見つかりました: ファイル {csv_path}, 行番号 {idx + 1}")

def main():
    # JSONファイルの読み込み
    file_path = [VAD_LIST + "/" + f for f in os.listdir(VAD_LIST)]
    for f in file_path:
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"エラー: JSONファイルの読み込みに失敗しました: {e}")
            return

        # 入力規則の検証
        if validate_structure(data):
            continue
        else:
            print(f"エラー: ファイルの構造が不正です: {f}")
            return
    print("全てのファイルが正常に読み込まれました。")

    # CSVファイルの内容を確認
    check_csv_content(CSV)

if __name__ == "__main__":
    main()