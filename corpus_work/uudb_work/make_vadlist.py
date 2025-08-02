import os
import re
import json
import numpy as np
from tqdm import tqdm

UUDB_DIR = "/autofs/diamond2/share/corpus/UUDB/UUDB/Sessions/"
VAD_LIST = "./vad_list"
temp = r'Utterance\s+UtteranceID="\d+"\s+Channel="([LR])"\s+UtteranceStartTime="(\d+\.\d+)"\s+UtteranceEndTime="(\d+\.\d+)"'

# 処理対象のディレクトリを事前にフィルタリング
target_dirs = [d for d in os.listdir(UUDB_DIR) if os.path.isdir(os.path.join(UUDB_DIR, d))]

# tqdmでディレクトリ処理の進捗を表示
for dir_name in tqdm(target_dirs, desc="Processing directories"):
    uudb_directories = []
    xml_file_paths = []
    even_result = []
    odd_result = []
    all_array = []
    output_file_name = ""
    
    xml_file_path = os.path.join(UUDB_DIR, dir_name, f"{dir_name}.xml")
    if os.path.exists(xml_file_path):
        xml_file_paths.append(xml_file_path)
        matches = re.findall(r'"([^"]+)"', temp)
        pattern = ",".join(matches)
        
        with open(os.path.join(xml_file_path), "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                match = re.search(temp, line)
                if match:
                    label = match.group(1)
                    start_time = float(match.group(2))
                    end_time = float(match.group(3))
                    if label == 'R':
                        even_result.append([start_time, end_time])
                    else:
                        odd_result.append([start_time, end_time])
            
            all_array = [odd_result, even_result]
            output_file_name = re.sub(r'/autofs/diamond2/share/corpus/UUDB/UUDB/Sessions/C0[0-6][0-9]/', '', xml_file_path)
            output_file_name = output_file_name.replace('.xml', '') + ".json"
            
            with open(os.path.join(VAD_LIST, output_file_name), "w") as json_file:
                json.dump(all_array, json_file)
    
print(f"処理が完了しました")