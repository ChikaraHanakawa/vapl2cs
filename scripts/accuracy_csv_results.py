import pandas as pd
import numpy as np
import os

def extract_bacc_p_now_value(file_path):
    """
    指定されたCSVファイルからbacc_p_nowの最大値とそのしきい値を抽出する
    """
    try:
        df = pd.read_csv(file_path)
        # 不要な空白などを削除
        df.columns = df.columns.str.strip()

        # 最大のbacc_p_nowの値とそのインデックスを取得
        max_bacc = df['bacc_p_now'].max()
        max_threshold = df.loc[df['bacc_p_now'].idxmax(), 'threshold']

        # フォーマット
        max_value_percent = max_bacc * 100
        formatted_value = f"{int(max_value_percent)}.{int(max_value_percent*10)%10}"

        return formatted_value
    
    except Exception as e:
        print(f"ファイル処理中にエラーが発生しました {file_path}: {e}")
        return "N/A"

def generate_bacc_p_now_results_table():
    """
    各モデル、評価データ、評価指標の組み合わせに対するbacc_p_now値を表形式で出力する
    """
    # モデル、評価データ、評価指標のリスト
    models = ['multi', 'without_tabidachi', 'without_cejc', 'without_waseda_soma']
    test_datasets = ['tabidachi', 'cejc', 'waseda_soma', 'csj', 'pasd', 'rwcp', 'uudb']
    metrics = ['hs', 'predhs', 'bc']
    
    # 各評価指標ごとに結果を格納するための辞書
    all_results = {}
    
    # 各評価指標についてDataFrameを作成
    for metric in metrics:
        # 結果を格納するDataFrameを作成
        results_df = pd.DataFrame(index=models, columns=test_datasets)
        
        # 各モデルと評価データの組み合わせを処理
        for model in models:
            for test_dataset in test_datasets:
                file_path = f'~/project/2025/test/StreamVAP/results/250503_results/{model}/{test_dataset}/{metric}/accuracy.csv'
                try:
                    # bacc_p_nowの値を抽出
                    results_df.loc[model, test_dataset] = extract_bacc_p_now_value(file_path)
                except Exception as e:
                    results_df.loc[model, test_dataset] = "N/A"
                    print(f"Error processing {model}/{test_dataset}/{metric}: {e}")
        
        # 結果をCSVファイルに保存
        output_path = f'~/project/2025/test/StreamVAP/results/250503_results/bacc_p_now_{metric}.csv'
        results_df.to_csv(output_path)
        print(f"bacc_p_now results for {metric} saved to {output_path}")
        
        # 全結果辞書に追加
        all_results[metric] = results_df
    
    return all_results

# メイン処理：bacc_p_now値を抽出してCSVファイルに保存
if __name__ == "__main__":
    results = generate_bacc_p_now_results_table()
    
    print("\nResults saved successfully!")
    
    # 各評価指標の結果のサンプルを表示
    for metric, df in results.items():
        print(f"\nSample of results for {metric}:")
        print(df.head())