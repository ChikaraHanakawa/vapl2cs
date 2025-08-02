import os
import re
import json
import numpy as np
from tqdm import tqdm

CSJ_DIR = "/autofs/diamond/share/corpus/CSJ/"
CSJ_TRN = os.path.join(CSJ_DIR, "TRN/Form1/noncore")
VAD_LIST = "./vad_list"
# 結果を格納するリスト
output_file_name = ""

def read_tnf():
    # 正規表現パターン
    pattern = r"(\d{4}) (\d+\.\d+)-(\d+\.\d+) ([RL]):"
    
    # 処理対象のファイル一覧を取得
    files = [f for f in os.listdir(CSJ_TRN) if f.startswith("D")]
    
    # tqdmでファイル処理の進捗を表示
    for filename in tqdm(files, desc="Processing files"):
        start_time = 0.0
        end_time = 0.0
        odd_result = []
        even_result = []
        all_array = []
        label = ""
        
        with open(os.path.join(CSJ_TRN, filename), "r", encoding="shift-jis") as file:
            lines = file.readlines()
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    start_time = float(match.group(2))
                    end_time = float(match.group(3))
                    label = match.group(4)
                    if label == "R":
                        even_result.append([start_time, end_time])
                    elif label == "L":
                        odd_result.append([start_time, end_time])
        
        all_array = [odd_result, even_result]
        # JSONファイルに書き出す
        output_file_name = filename.replace('.trn', '') + ".json"
        with open(os.path.join(VAD_LIST, output_file_name), "w") as json_file:
            json.dump(all_array, json_file)
    print(f"処理が完了しました")

read_tnf()