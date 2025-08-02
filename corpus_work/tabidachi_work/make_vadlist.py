import os
import json
import pandas as pd
from tqdm import tqdm

TABIDACHI_DIR = ["/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2019-1",
                 "/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2109-2",
                 "/autofs/diamond3/share/corpus/Tabidachi/extracted/Tabidachi2109-3",]
VAD_LIST = "./vad_list"

def time_to_sec(time):
    minutes, seconds = time.split(":")
    minutes = int(minutes) * 60
    time = minutes + float(seconds)
    return time

def write_json(import_file_name, all_array):
    file_name = import_file_name.replace('.tsv', '') + ".json"
    output_file_name = file_name[74:]
    with open(os.path.join(VAD_LIST, output_file_name), "w") as json_file:
        json.dump(all_array, json_file)

def read_tsv():
    file = []
    for directory in TABIDACHI_DIR:
        file.extend([directory + "/" + f + '/' + f + '_zoom.tsv' for f in os.listdir(directory) if f != ".DS_Store"])
    for f in tqdm(file):
        odd_result = []
        even_result = []
        output_result = []
        tsv = pd.read_csv(f, sep='\t')
        for line in range(tsv.shape[0]):
            speaker_name = tsv.iat[line, 0]
            if speaker_name is None or not isinstance(speaker_name, str):
                continue
            if "オペレータ" in speaker_name:
                start_time = time_to_sec(tsv.at[line, "発話開始時間"])
                end_time = time_to_sec(tsv.at[line, "発話終了時間"])
                even_result.append([start_time, end_time])
            elif "カスタマ" in speaker_name:
                start_time = time_to_sec(tsv.at[line, "発話開始時間"])
                end_time = time_to_sec(tsv.at[line, "発話終了時間"])
                odd_result.append([start_time, end_time])
        output_result = [odd_result, even_result]
        write_json(f, output_result)

read_tsv()