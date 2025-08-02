#!/usr/bin/env python3
# python scripts/fixed_ratio_selector.py input.csv output_directory --train 0.8 --val 0.1 --test 0.1

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

def create_datasets_with_ratio(input_csv, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    入力CSVから指定された割合でトレーニング・開発・評価データセットを作成する
    
    Args:
        input_csv (str): 入力CSVファイルのパス（duration列を含む）
        output_dir (str): 出力ディレクトリのパス
        train_ratio (float): トレーニングデータセットの割合（0～1）
        val_ratio (float): 開発データセットの割合（0～1）
        test_ratio (float): 評価データセットの割合（0～1）
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
    
    # 割合の合計が1.0になるか確認
    sum_ratio = train_ratio + val_ratio + test_ratio
    if abs(sum_ratio - 1.0) > 1e-5:  # 浮動小数点の誤差を考慮
        print(f"警告: 指定された割合の合計が1.0ではありません（合計: {sum_ratio}）", file=sys.stderr)
        # 割合を自動調整
        scale = 1.0 / sum_ratio
        train_ratio *= scale
        val_ratio *= scale
        test_ratio *= scale
        print(f"割合を自動調整しました: train={train_ratio:.3f}, val={val_ratio:.3f}, test={test_ratio:.3f}", file=sys.stderr)
    
    # データをランダムに並べ替え
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 全体の時間を計算
    total_duration = df['duration'].sum()
    
    # 各データセットの目標時間
    train_target = total_duration * train_ratio
    val_target = total_duration * val_ratio  # ← 変数名も変更
    test_target = total_duration * test_ratio

    # データセットを分割
    datasets = {
        "train": {"target": train_target, "data": []},
        "val": {"target": val_target, "data": []},
        "test": {"target": test_target, "data": []}
    }

    # データ分配処理の中で名前を合わせる
    current_dataset = "train"
    current_duration = 0

    for _, row in df.iterrows():
        if current_dataset == "train" and current_duration >= train_target:
            current_dataset = "val"
            current_duration = 0
        elif current_dataset == "val" and current_duration >= val_target:
            current_dataset = "test"
            current_duration = 0
        
        datasets[current_dataset]["data"].append(row)
        current_duration += row['duration']
    
    # 結果を保存
    for dataset_name, dataset_info in datasets.items():
        data = dataset_info["data"]
        if not data:
            print(f"警告: {dataset_name}データセットにデータがありません", file=sys.stderr)
            continue
        
        # DataFrameに変換
        dataset_df = pd.DataFrame(data)
        
        # 合計時間を計算
        actual_duration = dataset_df['duration'].sum()
        
        # 結果を出力CSVに書き込む
        hours, remainder = divmod(actual_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # ファイル名に実際の割合を含める
        actual_ratio = actual_duration / total_duration
        output_filename = f"{dataset_name}_{actual_ratio:.2f}.csv"
        output_filepath = os.path.join(output_dir, output_filename)
        
        try:
            dataset_df.to_csv(output_filepath, index=False)
        except Exception as e:
            print(f"エラー: CSVファイルの書き込みに失敗しました: {e}", file=sys.stderr)
            continue
        
        print(f"{dataset_name}データセット: {len(data)} 件のファイル、合計時間: {int(hours)}時間{int(minutes)}分{int(seconds)}秒 ({actual_duration:.2f}秒)")
        print(f"目標割合: {dataset_info['target'] / total_duration:.2f}, 実際の割合: {actual_ratio:.2f}")
        print(f"出力ファイル: {output_filepath}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='音声ファイルの再生時間を計測し、指定された割合でデータセットを分割するプログラム')
    parser.add_argument('input_csv', help='入力CSVファイルのパス（audio_path列を含む）')
    parser.add_argument('output_dir', help='出力ディレクトリのパス')
    parser.add_argument('--temp_csv', help='duration列を追加した一時CSVファイルのパス（省略時は自動生成）')
    parser.add_argument('--train', type=float, default=0.8, help='トレーニングデータセットの割合（0～1）、デフォルトは0.8')
    parser.add_argument('--val', type=float, default=0.1, help='開発データセットの割合（0～1）、デフォルトは0.1')
    parser.add_argument('--test', type=float, default=0.1, help='評価データセットの割合（0～1）、デフォルトは0.1')
    
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
    create_datasets_with_ratio(processed_csv, args.output_dir, 
                              args.train, args.val, args.test)

if __name__ == "__main__":
    main()