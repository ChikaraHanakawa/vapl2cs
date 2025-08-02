import csv
import os
from pydub import AudioSegment

# コーパスの配列
corpora = ["cejc", "uudb", "pasd", "csj", "rwcp", "waseda_soma", "tabidachi"]

# ベースとなるCSVファイルのディレクトリ
base_dir = '/home/hanakawa/project/2025/test/StreamVAP/data/splits_'

for corpus in corpora:
    # コーパスごとのCSVファイルパス
    csv_file = os.path.join(base_dir + corpus, 'audio_vad.csv')

    if not os.path.exists(csv_file):
        print(f"CSVファイルが見つかりません: {csv_file}")
        continue

    total_duration_ms = 0

    # CSVファイルを開く
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = row['audio_path']
            if audio_path and os.path.exists(audio_path):
                audio = AudioSegment.from_file(audio_path)
                total_duration_ms += len(audio)
            else:
                print(f"オーディオファイルが見つかりません: {audio_path}")

    # ミリ秒を秒に変換
    total_duration_sec = total_duration_ms / 1000

    # 秒を時間・分・秒に変換
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = int(total_duration_sec % 60)

    # 出力
    print(f"【{corpus}】")
    print(f"合計再生時間: {total_duration_sec:.2f}秒")
    print(f"合計再生時間: {hours}時間{minutes}分{seconds}秒")
    print("-" * 30)
