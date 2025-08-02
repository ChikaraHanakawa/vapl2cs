import pandas as pd
import numpy as np
import os

def extract_bacc_p_now_value(file_path):
    """
    指定されたCSVファイルからbacc_p_nowの最大値とそのしきい値を抽出する
    """
    try:
        if not os.path.exists(file_path):
            return "N/A"
            
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

def generate_results_for_metric(metric):
    """
    指定された評価指標(hs, bc, predhs)についてbacc_p_now値のテーブルを生成する
    """
    # ベースモデル、モデル、テストデータのリスト
    bases = ['tabidachi', 'cejc', 'waseda_soma']
    
    # 特殊なケース：ベースモデル自体もテストに含まれている
    test_mapping = {
        'tabidachi': ['cejc', 'uudb', 'pasd', 'csj', 'rwcp', 'waseda_soma', 'tabidachi'],
        'cejc': ['cejc', 'uudb', 'pasd', 'csj', 'rwcp', 'waseda_soma', 'tabidachi'],
        'waseda_soma': ['cejc', 'uudb', 'pasd', 'csj', 'rwcp', 'waseda_soma', 'tabidachi']
    }
    
    # 各ベースモデルが使用可能なモデルを定義
    # 各ベースモデルは自分自身をモデルとしても使用可能
    model_mapping = {
        'tabidachi': ['tabidachi', 'csj', 'rwcp', 'pasd', 'uudb'],
        'cejc': ['cejc', 'csj', 'rwcp', 'pasd', 'uudb'],
        'waseda_soma': ['waseda_soma', 'csj', 'rwcp', 'pasd', 'uudb']
    }
    
    # 全てのモデルと全てのテストケースをリストアップ
    all_models = sorted(list(set().union(*model_mapping.values())))
    all_tests = sorted(list(set().union(*test_mapping.values())))
    
    # マルチインデックスを使用: (ベースモデル, モデル)と(テスト)でクロステーブルに
    multi_index = []
    for base in bases:
        for model in model_mapping[base]:
            multi_index.append((base, model))
    
    multi_index = pd.MultiIndex.from_tuples(multi_index, names=['base', 'model'])
    results_df = pd.DataFrame(index=multi_index, columns=all_tests)
    
    # 各ベースモデル、モデル、テストの組み合わせを処理
    for base in bases:
        for model in model_mapping[base]:
            for test in test_mapping[base]:
                # 実際のファイルパスを構築
                accuracy_file = f"results/250501_results/{base}/{model}/{test}/{metric}/accuracy.csv"
                
                # bacc_p_nowの値を抽出
                try:
                    bacc_value = extract_bacc_p_now_value(accuracy_file)
                    results_df.loc[(base, model), test] = bacc_value
                except Exception as e:
                    results_df.loc[(base, model), test] = "N/A"
                    print(f"Error processing {base}/{model}/{test}/{metric}: {e}")
    
    # 結果をCSVファイルに保存
    output_dir = 'results/250501_results/bacc_p_now'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f'{output_dir}/bacc_p_now_{metric}_comparison.csv'
    results_df.to_csv(output_path)
    print(f"bacc_p_now results for {metric} saved to {output_path}")
    
    # 各ベースモデルごとの結果も個別に保存
    for base in bases:
        # 該当するベースモデルの行だけを抽出
        if base in results_df.index.get_level_values('base'):
            base_results = results_df.loc[base]
            base_output_path = f'{output_dir}/bacc_p_now_{metric}_{base}.csv'
            base_results.to_csv(base_output_path)
            print(f"Base model {base} results for {metric} saved to {base_output_path}")
    
    return results_df

def display_formatted_results(results_df, metric):
    """
    結果を見やすく整形して表示する
    """
    print(f"\n=== 各ベースモデルごとの{metric}のbacc_p_now最大値（%） ===")
    
    for base in results_df.index.levels[0]:
        print(f"\n【ベースモデル: {base}】")
        base_df = results_df.loc[base]
        # カラム名のスペースを揃えて表示
        print(base_df.fillna("N/A").to_string())

def generate_all_metrics_results():
    """
    全ての評価指標（hs, bc, predhs）に対して結果を生成する
    """
    metrics = ['hs', 'bc', 'predhs']
    results = {}
    
    for metric in metrics:
        print(f"\n処理中: 評価指標 {metric}")
        results[metric] = generate_results_for_metric(metric)
        display_formatted_results(results[metric], metric)
    
    # 全ての結果を一つのファイルにまとめる
    combined_results = {}
    for metric in metrics:
        # 各メトリックにプレフィックスをつけてカラム名を変更
        metric_results = results[metric].copy()
        renamed_columns = {col: f"{metric}_{col}" for col in metric_results.columns}
        metric_results.rename(columns=renamed_columns, inplace=True)
        combined_results[metric] = metric_results
    
    # 横方向に結合
    all_metrics_df = pd.concat(combined_results.values(), axis=1)
    
    # 結果を保存
    output_dir = 'results/250501_results/bacc_p_now'
    os.makedirs(output_dir, exist_ok=True)
    all_metrics_path = f'{output_dir}/bacc_p_now_all_metrics.csv'
    all_metrics_df.to_csv(all_metrics_path)
    print(f"\n全ての評価指標の結果を結合したファイルを保存しました: {all_metrics_path}")
    
    return results

# メイン処理：全評価指標のbacc_p_now値を抽出してCSVファイルに保存
if __name__ == "__main__":
    # 必要なディレクトリを作成
    os.makedirs('results', exist_ok=True)
    
    # 全ての評価指標の結果テーブルを生成
    all_results = generate_all_metrics_results()
    
    print("\nAll results saved successfully!")