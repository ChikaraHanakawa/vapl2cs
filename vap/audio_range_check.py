#!/usr/bin/env python

import torch
import torchaudio
import os
from pathlib import Path
from vap.utils.audio import load_waveform, get_audio_info, time_to_samples, sample_to_time

def analyze_audio_file(file_path, start_time=None, end_time=None, verbose=True):
    """
    指定されたファイルの音声情報を分析し、時間範囲の妥当性を確認します
    
    Args:
        file_path: 音声ファイルのパス
        start_time: 開始時間（秒）
        end_time: 終了時間（秒）
        verbose: 詳細な情報を出力するかどうか
    
    Returns:
        dict: 分析結果を含む辞書
    """
    # ファイルの存在確認
    if not os.path.exists(file_path):
        print(f"エラー: ファイルが存在しません: {file_path}")
        return None
    
    try:
        # ファイル情報を取得
        info = get_audio_info(file_path)
        
        if verbose:
            print(f"ファイル: {Path(file_path).name}")
            print(f"サンプリングレート: {info['sample_rate']} Hz")
            print(f"チャネル数: {info['num_channels']}")
            print(f"総サンプル数: {info['num_frames']}")
            print(f"総再生時間: {info['duration']:.2f} 秒")
        
        # 時間範囲のバリデーション
        file_duration = info['duration']
        max_time = file_duration
        
        if start_time is not None and end_time is not None:
            if start_time < 0:
                print(f"警告: 開始時間が0より小さいため、0に設定します")
                start_time = 0
            
            if end_time > max_time:
                print(f"警告: 終了時間がファイル長（{max_time:.2f}秒）を超えています")
                end_time = max_time
            
            if start_time >= end_time:
                print(f"エラー: 無効な時間範囲: 開始={start_time:.2f}秒, 終了={end_time:.2f}秒")
                print(f"       開始時間は終了時間より小さくなければなりません")
                return None
            
            # サンプル数に変換
            start_frame = time_to_samples(start_time, info["sample_rate"])
            end_frame = time_to_samples(end_time, info["sample_rate"])
            num_frames = end_frame - start_frame
            
            print(f"指定時間範囲: {start_time:.2f}秒 ～ {end_time:.2f}秒 (合計: {end_time - start_time:.2f}秒)")
            print(f"指定サンプル範囲: {start_frame} ～ {end_frame} (合計: {num_frames} サンプル)")
        
        # 波形を読み込み（エラーハンドリング付き）
        try:
            waveform, sr = load_waveform(file_path, sample_rate=None, 
                                         start_time=start_time, end_time=end_time)
            
            if waveform.numel() == 0 or waveform.shape[-1] == 0:
                print(f"エラー: 読み込まれた波形が空です")
                return None
            
            print(f"読み込み成功: 波形サイズ = {tuple(waveform.shape)}, サンプリングレート = {sr} Hz")
            
            # 実際に読み込まれたサンプル数から実際の時間を計算
            actual_duration = sample_to_time(waveform.shape[-1], sr)
            print(f"実際の長さ: {actual_duration:.2f}秒")
            
            return {
                "file_path": file_path,
                "waveform": waveform,
                "sample_rate": sr,
                "start_time": start_time,
                "end_time": end_time,
                "actual_duration": actual_duration,
                "shape": tuple(waveform.shape)
            }
            
        except Exception as e:
            print(f"波形読み込みエラー: {e}")
            return None
        
    except Exception as e:
        print(f"ファイル分析エラー: {e}")
        return None

def test_time_ranges(file_path):
    """
    様々な時間範囲でファイルの読み込みをテストします
    
    Args:
        file_path: 音声ファイルのパス
    """
    # ファイル情報を取得
    info = get_audio_info(file_path)
    max_time = info["duration"]
    
    print(f"\n=== ファイル全体 ===")
    analyze_audio_file(file_path, None, None)
    
    print(f"\n=== 最初の10秒 ===")
    analyze_audio_file(file_path, 0, 10)
    
    print(f"\n=== 中間の10秒 ===")
    mid_point = max_time / 2
    analyze_audio_file(file_path, mid_point - 5, mid_point + 5)
    
    print(f"\n=== 最後の10秒 ===")
    analyze_audio_file(file_path, max_time - 10, max_time)
    
    # 問題のあった時間範囲
    print(f"\n=== 問題の時間範囲 ===")
    analyze_audio_file(file_path, 1659.7099609375, 1679.7099609375)
    
    # 修正された時間範囲（ファイル長をチェックして調整）
    print(f"\n=== 修正された時間範囲 ===")
    adjusted_start = min(1659.7099609375, max_time - 20)
    adjusted_end = min(1679.7099609375, max_time)
    analyze_audio_file(file_path, adjusted_start, adjusted_end)

if __name__ == "__main__":
    file_path = "/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2109-3/307_3_3/307_3_3_zoom.wav"
    
    # ファイル情報の詳細分析
    print("\n=== ファイル情報分析 ===")
    analyze_audio_file(file_path, verbose=True)
    
    # 問題となっていた時間範囲をテスト
    print("\n=== 問題の時間範囲テスト ===")
    result = analyze_audio_file(file_path, 1659.7099609375, 1679.7099609375)
    
    if result:
        # 読み込みに成功した場合、波形を表示
        waveform = result["waveform"]
        
        # 波形の簡単な統計情報を表示
        print(f"波形の統計情報:")
        print(f"  最小値: {waveform.min().item():.4f}")
        print(f"  最大値: {waveform.max().item():.4f}")
        print(f"  平均値: {waveform.mean().item():.4f}")
        print(f"  標準偏差: {waveform.std().item():.4f}")
    
    # 他の時間範囲もテスト
    print("\n=== 様々な時間範囲のテスト ===")
    test_time_ranges(file_path)
