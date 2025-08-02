
# python scripts/fixed_duration_selector.py input.csv output_directory --dev 5 --test 10 --durations 10 20 30 40 50 60 70 80 90 100

import argparse
import csv
import sys
import random
import os
from pathlib import Path
import librosa
import tqdm
import pandas as pd

def get_audio_duration(audio_path):
    """
    音声ファイルの再生時間を取得する
    
    Args:
        audio_path (str): 音声ファイルのパス
        
    Returns:
        float: 音声ファイルの再生時間（秒）
    """
    try:
        # librosaを使用して音声ファイルの再生時間を取得
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception as e:
        print(f"警告: ファイル '{audio_path}' の再生時間取得に失敗しました: {e}", file=sys.stderr)
        return None

def add_duration_to_csv(input_csv, output_csv=None):
    """
    CSVファイルにduration列を追加する
    
    Args:
        input_csv (str): 入力CSVファイルのパス
        output_csv (str, optional): 出力CSVファイルのパス。省略時は入力ファイルを上書き
        
    Returns:
        str: 処理後のCSVファイルのパス
    """
    if output_csv is None:
        output_csv = input_csv
    
    # CSVを読み込む
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        return None
    
    # audio_path列の確認
    if 'audio_path' not in df.columns:
        print("エラー: 'audio_path'列が見つかりません。", file=sys.stderr)
        return None
    
    # 進捗表示付きで各音声ファイルの再生時間を取得
    print("音声ファイルの再生時間を計測中...")
    durations = []
    
    for path in tqdm.tqdm(df['audio_path']):
        duration = get_audio_duration(path)
        durations.append(duration)
    
    # duration列を追加または更新
    df['duration'] = durations
    
    # Noneの値を持つ行（再生時間の取得に失敗した行）を削除
    df = df.dropna(subset=['duration'])
    
    # CSVに保存
    try:
        df.to_csv(output_csv, index=False)
        print(f"再生時間を追加したCSVを保存しました: {output_csv}")
        return output_csv
    except Exception as e:
        print(f"エラー: CSVファイルの書き込みに失敗しました: {e}", file=sys.stderr)
        return None

def create_datasets(input_csv, output_dir, durations_hours=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dev_hours=5, test_hours=10):
    """
    入力CSVから開発・評価データと各トレーニングデータセットを作成する
    トレーニングデータ同士は重複してもよいが、開発・評価データとトレーニングデータは重複しない
    
    Args:
        input_csv (str): 入力CSVファイルのパス（duration列を含む）
        output_dir (str): 出力ディレクトリのパス
        durations_hours (list): 作成する各トレーニングデータセットの時間（時間単位）
        dev_hours (int): 開発データセットの時間（時間単位）
        test_hours (int): 評価データセットの時間（時間単位）
    """
    # 入力ファイルの確認
    if not Path(input_csv).exists():
        print(f"エラー: 入力ファイル '{input_csv}' が見つかりません。", file=sys.stderr)
        return False
    
    # 出力ディレクトリの確認と作成
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    # CSVを読み込む
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        return False
    
    # duration列の確認
    if 'duration' not in df.columns:
        print("エラー: 'duration'列が見つかりません。", file=sys.stderr)
        return False
    
    # データをランダムに並べ替え
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 最初に開発・評価データ用のデータを選別
    dev_test_pool = df.copy()
    dev_test_indices = set()  # 開発・評価データに使用するインデックス
    
    # 開発データセットと評価データセットを作成
    dev_test_datasets = [
        ("dev", dev_hours * 3600),
        ("test", test_hours * 3600)
    ]
    
    for dataset_name, target_seconds in dev_test_datasets:
        selected_indices = []
        total_duration = 0.0
        
        for i, (_, row) in enumerate(dev_test_pool.iterrows()):
            if i in dev_test_indices:
                continue
                
            duration = row['duration']
            if total_duration + duration <= target_seconds:
                selected_indices.append(i)
                dev_test_indices.add(i)
                total_duration += duration
            elif abs(target_seconds - total_duration) > abs(target_seconds - (total_duration + duration)):
                # 目標時間に近づく場合は追加
                selected_indices.append(i)
                dev_test_indices.add(i)
                total_duration += duration
                break
        
        # 選択した行を取得
        selected_df = dev_test_pool.iloc[selected_indices]
        
        # 結果を出力CSVに書き込む
        dataset_filename = f"{dataset_name}_{int(total_duration // 3600)}h.csv"
        output_filepath = os.path.join(output_dir, dataset_filename)
        
        try:
            selected_df.to_csv(output_filepath, index=False)
        except Exception as e:
            print(f"エラー: CSVファイルの書き込みに失敗しました: {e}", file=sys.stderr)
            continue
        
        hours, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{dataset_name}データセット: {len(selected_indices)} 件のファイル、合計時間: {int(hours)}時間{int(minutes)}分{int(seconds)}秒 ({total_duration:.2f}秒)")
        print(f"出力ファイル: {output_filepath}")
    
    # 開発・評価データを除外してトレーニングデータ用のプールを作成
    train_pool = df.drop(df.index[list(dev_test_indices)]).reset_index(drop=True)
    
    # 各時間ごとのトレーニングデータセットを作成（重複可）
    for hours in durations_hours:
        target_seconds = hours * 3600
        selected_indices = []
        total_duration = 0.0
        
        # 必要に応じてシャッフル（各データセットで別の順序にする場合）
        train_pool_shuffled = train_pool.sample(frac=1, random_state=hours).reset_index(drop=True)
        
        for i, (_, row) in enumerate(train_pool_shuffled.iterrows()):
            duration = row['duration']
            if total_duration + duration <= target_seconds:
                selected_indices.append(i)
                total_duration += duration
            elif abs(target_seconds - total_duration) > abs(target_seconds - (total_duration + duration)):
                # 目標時間に近づく場合は追加
                selected_indices.append(i)
                total_duration += duration
                break
        
        # 選択した行を取得
        selected_df = train_pool_shuffled.iloc[selected_indices]
        
        # 結果を出力CSVに書き込む
        output_filename = f"train_{hours}h.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        
        try:
            selected_df.to_csv(output_filepath, index=False)
        except Exception as e:
            print(f"エラー: CSVファイルの書き込みに失敗しました: {e}", file=sys.stderr)
            continue
        
        hours_actual, remainder = divmod(total_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"trainデータセット {hours}時間: {len(selected_indices)} 件のファイル、合計時間: {int(hours_actual)}時間{int(minutes)}分{int(seconds)}秒 ({total_duration:.2f}秒)")
        print(f"出力ファイル: {output_filepath}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='音声ファイルの再生時間を計測し、指定された時間のデータセットを作成するプログラム')
    parser.add_argument('input_csv', help='入力CSVファイルのパス（audio_path列を含む）')
    parser.add_argument('output_dir', help='出力ディレクトリのパス')
    parser.add_argument('--temp_csv', help='duration列を追加した一時CSVファイルのパス（省略時は自動生成）')
    parser.add_argument('--dev', type=float, default=5, help='開発データセットの時間（時間単位）、デフォルトは5時間')
    parser.add_argument('--test', type=float, default=10, help='評価データセットの時間（時間単位）、デフォルトは10時間')
    parser.add_argument('--durations', type=int, nargs='+', default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                        help='トレーニングデータセットの時間（時間単位）、例: --durations 10 20 30 40 50 60 70 80 90 100')
    
    args = parser.parse_args()
    
    # 一時CSVファイルのパスを決定
    temp_csv = args.temp_csv
    if temp_csv is None:
        input_path = Path(args.input_csv)
        temp_csv = str(input_path.parent / f"{input_path.stem}_with_duration{input_path.suffix}")
    
    # 音声ファイルの再生時間を計測してCSVに追加
    processed_csv = add_duration_to_csv(args.input_csv, temp_csv)
    if processed_csv is None:
        return
    
    # データセットを作成
    create_datasets(processed_csv, args.output_dir, args.durations, args.dev, args.test)

if __name__ == "__main__":
    main()