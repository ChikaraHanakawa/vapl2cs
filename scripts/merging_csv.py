import os
import pandas as pd

def merge_csv_files(base_file_path, add_file_path, output_file_path):
    """
    2つのCSVファイルを結合し、新しいCSVファイルとして保存する
    
    Parameters:
    base_file_path (str): ベースとなるCSVファイルのパス
    add_file_path (str): 追加するCSVファイルのパス
    output_file_path (str): 出力先のCSVファイルのパス
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    try:
        # CSVファイルを読み込む
        base_df = pd.read_csv(base_file_path)
        add_df = pd.read_csv(add_file_path)
        
        # 列名の確認
        if list(base_df.columns) != list(add_df.columns):
            print(f"警告: {base_file_path}と{add_file_path}の列が一致しません")
            print(f"Base columns: {base_df.columns}")
            print(f"Add columns: {add_df.columns}")
        
        # DataFrameを結合
        combined_df = pd.concat([base_df, add_df], ignore_index=True)
        
        # 結合したDataFrameをCSVとして保存
        combined_df.to_csv(output_file_path, index=False)
        
        print(f"結合完了: {output_file_path}")
        print(f"  - {base_file_path} ({len(base_df)}行)")
        print(f"  - {add_file_path} ({len(add_df)}行)")
        print(f"  = 合計 {len(combined_df)}行")
        
    except Exception as e:
        print(f"エラー: {e}")

def main():
    # 基本ディレクトリとファイル情報
    base_path = "/home/hanakawa/project/2025/test/StreamVAP/data"
    file_types = ["train.csv", "val.csv"]
    
    # 各ディレクトリ設定
    base_directories = ["tabidachi", "cejc", "waseda_soma"]
    add_directories = ["csj", "rwcp", "pasd", "uudb"]
    
    # 各組み合わせでCSVファイルを結合
    for base_dir in base_directories:
        for add_dir in add_directories:
            for file_type in file_types:
                # 入力ファイルパス
                base_file = f"{base_path}/splits_{base_dir}/0430_exp/sliding_window_dset/{file_type}"
                add_file = f"{base_path}/splits_{add_dir}/0430_exp/sliding_window_dset/{file_type}"
                
                # 出力ファイルパス
                output_dir = f"{base_path}/splits_{base_dir}/0430_exp/sliding_window_dset/{add_dir}_add"
                output_file = f"{output_dir}/{file_type}"
                
                # ファイルが存在するか確認
                if not os.path.exists(base_file):
                    print(f"エラー: {base_file} が見つかりません")
                    continue
                
                if not os.path.exists(add_file):
                    print(f"エラー: {add_file} が見つかりません")
                    continue
                
                # CSVファイルを結合
                merge_csv_files(base_file, add_file, output_file)

if __name__ == "__main__":
    main()