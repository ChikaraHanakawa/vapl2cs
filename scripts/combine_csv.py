import pandas as pd
import os

def ensure_dir(directory):
    """指定されたディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def get_input_path(dataset, file_type):
    """データセットと種類（train/val）に基づいて入力ファイルのパスを返す"""
    return f"data/splits_{dataset}/0430_exp/sliding_window_dset/{file_type}.csv"

def get_output_path(output_type, dataset=None, file_type=None):
    """出力ファイルのパスを返す
    
    Args:
        output_type (str): 'multi' または 'without'
        dataset (str, optional): 'without' の場合に除外するデータセット名
        file_type (str, optional): 'train' または 'val'
    """
    if output_type == 'multi':
        return f"data/splits_multi/sliding_window_dset/{file_type}.csv"
    else:  # without
        return f"data/without_multi/{dataset}_without_{file_type}.csv"

def combine_datasets(datasets, output_type, excluded=None):
    """
    指定されたデータセットのCSVファイルを結合し、新しいCSVファイルを作成する
    
    Args:
        datasets (list): 結合するデータセットの名前リスト
        output_type (str): 出力タイプ ('multi' または 'without')
        excluded (str, optional): 除外されたデータセット名（'without'の場合）
    """
    # 訓練データ用とバリデーション用の結合データフレームを初期化
    combined_train = pd.DataFrame()
    combined_val = pd.DataFrame()
    
    # 各データセットのCSVファイルを読み込んで結合
    for dataset in datasets:
        # ファイルパスを指定
        train_path = get_input_path(dataset, "train")
        val_path = get_input_path(dataset, "val")
        
        # ファイルが存在する場合のみ読み込む
        if os.path.exists(train_path):
            df_train = pd.read_csv(train_path)
            # データソースを識別するための列を追加
            df_train['source'] = dataset
            # 結合する
            combined_train = pd.concat([combined_train, df_train], ignore_index=True)
            print(f"Added {len(df_train)} rows from {train_path}")
        else:
            print(f"Warning: {train_path} not found")
        
        if os.path.exists(val_path):
            df_val = pd.read_csv(val_path)
            # データソースを識別するための列を追加
            df_val['source'] = dataset
            # 結合する
            combined_val = pd.concat([combined_val, df_val], ignore_index=True)
            print(f"Added {len(df_val)} rows from {val_path}")
        else:
            print(f"Warning: {val_path} not found")
    
    # 出力ディレクトリを確保
    if output_type == 'multi':
        output_dir = os.path.dirname(get_output_path('multi', file_type='train'))
        ensure_dir(output_dir)
    else:
        output_dir = os.path.dirname(get_output_path('without', excluded, 'train'))
        ensure_dir(output_dir)
    
    # 結合したデータを保存
    if not combined_train.empty:
        output_train_path = get_output_path(output_type, excluded, 'train')
        combined_train.to_csv(output_train_path, index=False)
        print(f"Created {output_train_path} with {len(combined_train)} rows")
    
    if not combined_val.empty:
        output_val_path = get_output_path(output_type, excluded, 'val')
        combined_val.to_csv(output_val_path, index=False)
        print(f"Created {output_val_path} with {len(combined_val)} rows")

def main():
    # データセットのリスト
    all_datasets = ["tabidachi", "cejc", "waseda_soma"]
    
    # すべてのデータセットを結合した「multi」を作成
    combine_datasets(all_datasets, 'multi')
    
    # 各データセットを1つずつ除いた組み合わせを作成
    for exclude_dataset in all_datasets:
        # 除外するデータセット以外のリストを作成
        included_datasets = [d for d in all_datasets if d != exclude_dataset]
        # データセットを結合
        combine_datasets(included_datasets, 'without', exclude_dataset)

if __name__ == "__main__":
    main()