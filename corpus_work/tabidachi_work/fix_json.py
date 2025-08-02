import json
import os
import pandas as pd

VAD_LIST = "./vad_list"
FILENAME = "315_1_2_zoom.json"

path = os.path.join(VAD_LIST, FILENAME)
# JSONファイルを読み込む
with open(path, 'r') as file:
    data = json.load(file)

# すべての値から60秒を引く
for segment in data:
    for pair in segment:
        pair[0] -= 60
        pair[1] -= 60

# 変更されたデータをJSONファイルに書き込む
with open(path, 'w') as file:
    json.dump(data, file, indent=4)

print("すべての値から60秒を引きました。")