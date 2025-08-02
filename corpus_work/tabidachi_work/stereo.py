import os
from pydub import AudioSegment

# 対象の親ディレクトリ
base_dirs = [
    "/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2019-1",
    "/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2109-2",
    "/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2109-3",
]

# 出力先ディレクトリ
output_base_dir = "wav"
os.makedirs(output_base_dir, exist_ok=True)

# ステレオ結合と保存
def process_directory(dir_path):
    for root, dirs, files in os.walk(dir_path):
        # operator.m4a と user.m4a を検索
        operator_file = None
        user_file = None
        
        # ディレクトリ名を取得
        dir_name = os.path.basename(root)
        
        for file in files:
            if "_operator.m4a" in file:
                operator_file = os.path.join(root, file)
            elif "_user.m4a" in file:
                user_file = os.path.join(root, file)
        
        # 両方のファイルが見つかった場合にステレオ結合
        if operator_file and user_file:
            try:
                print(f"処理中: {dir_name}")
                print(f"  オペレーターファイル: {operator_file}")
                print(f"  ユーザーファイル: {user_file}")
                
                # 音声を読み込む
                operator_audio = AudioSegment.from_file(operator_file, format="m4a")
                user_audio = AudioSegment.from_file(user_file, format="m4a")
                
                # 長さを揃える
                max_length = max(len(operator_audio), len(user_audio))
                
                if len(operator_audio) < max_length:
                    operator_audio = operator_audio + AudioSegment.silent(duration=max_length - len(operator_audio))
                if len(user_audio) < max_length:
                    user_audio = user_audio + AudioSegment.silent(duration=max_length - len(user_audio))
                
                # ステレオチャンネルに分割
                # 注意: pydubでは、from_mono_audiosegmentsは両方のチャンネルを同じ音量で結合するので
                # 手動でステレオファイルを作成
                
                # まず両方をステレオに変換 (mono → stereo)
                stereo_operator = operator_audio.set_channels(2)
                stereo_user = user_audio.set_channels(2)
                
                # 左チャンネルだけ残す (右チャンネルの音量を0に)
                left_channel = stereo_operator.pan(-1.0)
                
                # 右チャンネルだけ残す (左チャンネルの音量を0に)
                right_channel = stereo_user.pan(1.0)
                
                # オーバーレイして結合 (加算合成)
                stereo_audio = left_channel.overlay(right_channel)
                
                # 出力パスを設定 (.wavに変更)
                output_filename = f"{dir_name}_zoom.wav"  # 拡張子をwavに
                output_path = os.path.join(output_base_dir, output_filename)
                
                # ファイルをエクスポート
                stereo_audio.export(output_path, format="wav")  # 形式をwavに
                print(f"  ステレオファイル作成成功: {output_path}")
            
            except Exception as e:
                print(f"  エラー発生: {str(e)}")

# メイン処理
print("処理を開始します...")
for base_dir in base_dirs:
    print(f"ディレクトリ処理中: {base_dir}")
    process_directory(base_dir)
print("処理が完了しました")